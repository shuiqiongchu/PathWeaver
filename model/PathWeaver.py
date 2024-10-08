import time
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

writer = SummaryWriter("logs")
t = time.time()

# 数据加载
class DataSource(Dataset):
    def __init__(self, data):
        self.data = data  # 存储数据

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, idx):
        sample = self.data[idx]  # 获取样本
        inputs = sample[:4]  # 前四个作为输入
        labels = sample[4:]  # 后两个作为标签
        return {'inputs': inputs, 'labels': labels}  # 返回字典形式

# 神经网络
class NerveW(nn.Module):
    def __init__(self):
        super(NerveW, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input):
        return self.model(input)

# 数据预处理
def load_data(file_name):
    dataSet = []
    with open(file_name, 'r') as file:
        data = file.readlines()  # 读取所有行
        for line in data:
            if line.startswith('P'):  # 判断当前行的第一个字符
                tl = line.replace("P", "").replace("\n", "")
                values = tl.split(":")
                values = [float(value) for value in values]  # 转换为数值类型
                dataSet.append(values)
    return dataSet

# 加载训练和测试数据
train_data = load_data('mouse_events.txt')
test_data = load_data('mouse_events1.txt')

# 转换为 tensor
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = DataSource(train_tensor)
test_dataset = DataSource(test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 标准化训练集
train_tensor = (train_tensor - train_tensor.mean(dim=0)) / train_tensor.std(dim=0)
train_tensor = train_tensor.cuda()

# 训练
n = NerveW()
n = n.cuda()

loss_fn = nn.MSELoss()
learning_rate = 0.001  # 调整学习率为更小的值

optimizer = torch.optim.Adam(n.parameters(), lr=learning_rate)  # 使用 Adam 优化器
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

# 对目标值进行归一化
def normalize_labels(labels):
    max_time = 5000
    max_points = 9999
    labels[:, 0] /= max_time
    labels[:, 1] /= max_points
    return labels

# 反归一化，用于模型评估时恢复原始单位
def denormalize_labels(labels):
    max_time = 5000
    max_points = 1000
    labels[:, 0] *= max_time
    labels[:, 1] *= max_points
    return labels

# 训练轮数
for i in range(10000):
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    train_loss = 0
    for batch in train_dataloader:
        inputs = batch['inputs'].cuda()
        tar = batch['labels'].cuda()

        # 归一化标签
        tar = normalize_labels(tar)

        # 进行前向传播
        opt = n(inputs)
        loss = loss_fn(opt, tar)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 更新学习率调度器
    avg_train_loss = train_loss / len(train_dataloader)
    scheduler.step(avg_train_loss)

    # 每100轮输出训练和测试结果
    if i % 100 == 0:
        print(f"第{i}轮训练损失: {avg_train_loss}")
        writer.add_scalar(f"训练损失{t}", avg_train_loss, i)

        test_loss = 0
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_inputs = test_batch['inputs'].cuda()
                test_tar = test_batch['labels'].cuda()

                # 归一化标签
                test_tar = normalize_labels(test_tar)

                test_opt = n(test_inputs)
                test_loss += loss_fn(test_opt, test_tar).item()

        avg_test_loss = test_loss / len(test_dataloader)
        print(f"第{i}轮测试损失: {avg_test_loss}")
        writer.add_scalar(f"测试损失{t}", avg_test_loss, i)
        torch.save(n.state_dict(), f'./simpleModel/model_epoch_{i}.pth')
        print(f'保存模型到 model_epoch_{i}.pth')

# 关闭 SummaryWriter
writer.close()
