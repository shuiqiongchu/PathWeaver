import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

# 创建 TensorBoard 实例
writer = SummaryWriter("logs")

# 自定义数据集
class PathDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 输入数据包含 (start_x, start_y, end_x, end_y, distance, avg_speed, total_time, point_idx, delay, next_point_x)
        inputs = [torch.tensor(row[:10], dtype=torch.float32) for row in sample]  # 每个时间步的输入
        # 输出数据包含 (delay, point_x, point_y)
        labels = [torch.tensor(row[8:], dtype=torch.float32) for row in sample]  # delay, point_x, point_y
        return inputs, labels

# 自定义 collate 函数以处理变长序列
def collate_fn(batch):
    inputs, labels = zip(*batch)  # 解压输入和标签
    inputs_padded = pad_sequence([pad_sequence(seq, batch_first=True) for seq in inputs], batch_first=True)  # 填充输入
    labels_padded = pad_sequence([pad_sequence(seq, batch_first=True) for seq in labels], batch_first=True)  # 填充标签
    return {'inputs': inputs_padded, 'labels': labels_padded}

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=4, num_layers=2):  # 输出维度修改为4
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 修改输出层
        self.fc = nn.Linear(hidden_size, output_size)  # 修改输出为4个

    def forward(self, x):
        # 初始化 LSTM 隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 取每个时间步的输出
        out = self.fc(out)
        return out

# 自定义损失函数
def custom_loss(output, target, weight_position=1.0, weight_time=0.5):
    # 分别计算 x, y 位置和时间延迟的损失
    position_loss = F.mse_loss(output[:, :, 1:], target[:, :, 1:])  # x, y坐标
    time_loss = F.mse_loss(output[:, :, 0], target[:, :, 0])  # 延迟时间
    return weight_position * position_loss + weight_time * time_loss
# 数据加载
def load_data(file_name):
    data = []
    datas = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.startswith('P'):
                values = line.strip().split(':')
                values = [float(v) for v in values]
                data.append(values)
            else:
                tl = line.replace("P", "").replace("\n", "")
                values2 = tl.split(":")
                values2 = [float(value) for value in values2]  # 转换为数值类型
                prefix_values = values2[:7]
                data_with_values = [prefix_values + row for row in data]
                datas.append(data_with_values)
                data.clear()

    return datas

# 加载数据
train_data = load_data('../dataAcquisition/mouse_events.txt')
test_data = load_data('../dataAcquisition/mouse_events1.txt')

# 创建数据集和数据加载器
train_dataset = PathDataset(train_data)
test_dataset = PathDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 创建模型实例
model = LSTMModel()
model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(model.device)

# 模型训练
# 模型训练
def train_lstm(model, train_loader, test_loader, num_epochs=20, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(model.device)
            targets = batch['labels'].to(model.device)
            outputs = model(inputs)

            # 确保 outputs 和 targets 在相同的形状
            if outputs.size(1) != targets.size(1):
                outputs = outputs[:, :targets.size(1), :]  # 裁剪 outputs

            loss = custom_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 记录训练损失到 TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        # 在测试集上评估模型
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for test_batch in test_loader:
                test_inputs = test_batch['inputs'].to(model.device)
                test_targets = test_batch['labels'].to(model.device)
                test_outputs = model(test_inputs)

                if test_outputs.size(1) != test_targets.size(1):
                    test_outputs = test_outputs[:, :test_targets.size(1), :]  # 裁剪 outputs

                test_loss = custom_loss(test_outputs, test_targets)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)

        # 合并图表
        writer.add_scalars("损失曲线", {
            '训练': avg_loss,
            '测试': avg_test_loss,
        }, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # 保存最优模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")

        # 每 10 轮保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path = f"./saveModel/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1}")

    writer.close()

# 开始训练
train_lstm(model, train_loader, test_loader, num_epochs=100000, lr=0.001)
