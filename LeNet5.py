#LeNet-5
#15epoch，优化器使用Adam，交叉熵损失函数，lr=0.001，初步准确率为63%
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5,stride=1,padding=0)
        self.conv2 = nn.Conv2d(6,16,5,stride=1,padding=0)
        self.conv3 = nn.Conv2d(16,120,5,stride=1,padding=0)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        out = self.fc3(x)
        return out

