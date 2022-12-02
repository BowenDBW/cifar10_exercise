import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

import network

# 设置 transforms
transform = transforms.Compose  ([
    transforms.ToTensor(),  # Numpy -> Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化 -1 ~ 1
])

# 下载数据集
# 训练集
train_set = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
# 测试集
test_set = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

# 批量读取数据,一次读一定数量的图片，数量大小取决于 GPU 能力
# 128 = RTX3080
BATCH_SIZE = 16

# 最后两个参数用于提高 GPU 使用率
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

# 可视化显示
classes = {
    "plane", "car", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck"
}


def imshow(img):
    img = img / 2 + 0.5  # 逆正则化
    np_img = img.numpy()  # tensor -> numpy
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # 改变通道顺序
    plt.show()


# 随机获取一批数据
imgs, labs = next(iter(train_loader))

print(imgs.shape)
print(labs.shape)
# 调用方法
imshow(torchvision.utils.make_grid(imgs))

# 输出这批图片对应的标签
print(' '.join('%5s' % classes[labs[i]] for i in range(BATCH_SIZE)))

net = network.Net().to('cuda')

criterion = nn.CrossEntropyLoss()  # 交叉式损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 优化器, 学习率，动量

EPOCHS = 200  # 学习轮数

for epoch in range(EPOCHS):

    train_loss = 0.0  # 用于每一轮的损失变化
    for i, (datas, labels) in enumerate(train_loader):
        datas, labels = datas.to('cuda'), labels.to('cuda')
        # 梯度置零
        optimizer.zero_grad()
        # 训练
        outputs = net(datas)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 累计损失
        train_loss += loss.item()
    # 打印信息
    print("Epoch: %d , Batch: %d, Loss: %.3f" % (epoch + 1, i + 1, train_loss/len(train_loader.dataset)))

PATH = './cifar_net_path'
torch.save(net.state_dict(), PATH)


# 测试
model = network.Net()
model.load_state_dict(torch.load(PATH))
correct = 0
total = 0
flag = True
with torch.no_grad():
    for i, (datas, labels) in enumerate(test_loader):
        # 输出
        outputs = model(datas)  # outputs.data.shape -> torch.Size([120,10])
        _, predicted = torch.max(outputs.data, dim=1)  # 第一个是值的张量，第二个是序号的张量
        # 累计数据量
        total += labels.size(0)  # labels.size() -> torch.Size([16]), labels.size(0) -> 16
        # 比较有多少个预测正确
        correct += predicted == labels.sum()  # 相同为 1，不同为 0，利用 sum 求和
    print("准确率" + format(correct / total * 100))


# 显示每一类预测的正确率
class_correct = list(0. for i in range[10])
total = list(0. for i in range[10])

with torch.no_grad():
    for (images, labels) in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)  # 获取每一行最大的索引
        c = (predicted == labels).squeeze()  # squeeze() 去掉 0 维的默认，unsqueeze 增加 1 维
        if labels.shape[0] == 128:
            for i in range[BATCH_SIZE]:
                label = labels[i]  # 获取每一个 label
                class_correct[label] += c[i].item()
                total[label] += 1

for i in range(10):
    print("正确率：%5s : %2d %%" % (classes[i], 100 * class_correct[i] / total[i]))
