import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time

# SummaryWriter 用于记录损失曲线和其他信息
writer = SummaryWriter("logs")
t = time.time()

# 数据加载函数
def load_data(file_name):
    dataSet = []
    with open(file_name, 'r') as file:
        data = file.readlines()  # 读取所有行
        for line in data:
            if line.startswith('P'):  # 判断当前行的第一个字符是否是 P
                tl = line.replace("P", "").replace("\n", "")
                values = tl.split(":")
                values = [float(value) for value in values]  # 转换为数值类型
                dataSet.append(values)
    return dataSet

# 自定义数据集
class DataSource(Dataset):
    def __init__(self, data):
        self.data = data  # 存储数据

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, idx):
        sample = self.data[idx]  # 获取样本
        inputs = sample[:6]  # 前六个作为输入
        labels = sample[6:]  # 后面的作为标签
        return {'inputs': inputs, 'labels': labels}  # 返回字典形式

# 神经网络模型
class PathEstimator(nn.Module):
    def __init__(self):
        super(PathEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 64),  # 输入 6 个特征
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出 2 个值：时间和路径点数量
        )

    def forward(self, input):
        return self.model(input)

# 自定义损失函数
def custom_loss(output, target, weight_time=1, weight_count=1.0):
    # time_loss: 总时间的损失，默认权重较小
    time_loss = nn.MSELoss()(output[:, 0], target[:, 0])

    # count_loss: 路径点数量的损失，默认权重较大
    count_loss = nn.MSELoss()(output[:, 1], target[:, 1])

    # 总损失
    return weight_time * time_loss + weight_count * count_loss


# 数据加载
train_data = load_data('../dataAcquisition/mouse_events.txt')
test_data = load_data('../dataAcquisition/mouse_events1.txt')

# 转换为 tensor
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# 数据标准化
train_tensor = (train_tensor - train_tensor.mean(dim=0)) / train_tensor.std(dim=0)
test_tensor = (test_tensor - test_tensor.mean(dim=0)) / test_tensor.std(dim=0)

# 创建数据集和数据加载器
train_dataset = DataSource(train_tensor)
test_dataset = DataSource(test_tensor)

print(len(train_dataset))
print(len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练和验证循环
def train_model(model, train_loader, test_loader, num_epochs=3, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    loss_fn = custom_loss  # 自定义损失函数
    model = model.cuda()  # 将模型放到 GPU
    best_loss = float('inf')  # 用于保存最好的模型
    step = 0  # 记录步数

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # 获取数据并转到 GPU
            inputs, targets = batch['inputs'].cuda(), batch['labels'].cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()

            # 每次反向传播调整完网络参数后，立即计算训练和测试集损失，并保存模型
            train_loss = loss.item()

            # 验证阶段
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for test_batch in test_loader:
                    test_inputs, test_targets = test_batch['inputs'].cuda(), test_batch['labels'].cuda()
                    test_outputs = model(test_inputs)
                    test_loss += loss_fn(test_outputs, test_targets).item()

            avg_test_loss = test_loss / len(test_loader)

            # 记录损失到 TensorBoard
            writer.add_scalars("损失曲线", {
                '训练': train_loss,
                '测试': avg_test_loss,
            }, step)

            # 每次调整后保存模型
            #torch.save(model.state_dict(), f'./saveModel/model_step_{step}.pth')
            print(f'保存模型到 model_step_{step}.pth')

            # 保存最好的模型
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(model.state_dict(), 'saveModel/best_model.pth')
                print(f"保存最佳模型到 best_model.pth，测试损失: {best_loss}")

            step += 1  # 更新步数

        # 更新学习率调度器
        scheduler.step(avg_test_loss)

    writer.close()

# 实例化模型并训练
model = PathEstimator()
train_model(model, train_dataloader, test_dataloader, num_epochs=100, lr=0.001)
