import torch.nn as nn
import torch.nn.functional as F

"""
知识点：
1. 特征图尺寸计算公式：[(原图片尺寸 - 卷积核尺寸) / 步长] + 1
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1 输入是 32 x 32 x 3, 计算 （32 - 5）/ 1 + 1 = 28, 那么通过conv1输出的是28 x 28 x 6
        self.conv1 = nn.Conv2d(3, 6, 5)  # input:3, output:6, kernel:5
        # 池化层 输入是 28 x 28 x 6, 窗口 2 x 2, 计算 28 / 2 = 14, 那么输出 14 x 14 x 6
        self.pool = nn.MaxPool2d(2, 2)  # kernel:2, stride:2
        # 卷积层2 输入是 14 x 14 x 6, 计算 （14 - 5）/ 1 + 1 = 10, 输出是 10 x 10 x 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input 16 x 5 x 5, output: 120
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)  # input 120, output: 84
        # 全连接层3
        self.fc3 = nn.Linear(84, 10)  # input 84, output: 10

    def forward(self, x):
        # 卷积 1
        """
        32 x 32 x 3 -> 28 x 28 x 6 -> 14 x 14 x 6
        """
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积 1
        """
        14 x 14 x 6 -> 10 x 10 x 16 -> 5 x 5 x 16
        """
        x = self.pool(F.relu(self.conv2(x)))
        # 改变 shape：拉平
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层 1
        x = F.relu(self.fc1(x))
        # 全连接层 2
        x = F.relu(self.fc2(x))
        # 全连接层 3
        x = F.relu(self.fc3(x))
        return x

