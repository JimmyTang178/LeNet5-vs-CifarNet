#15epoch,CrossEntropyloss=0.09,卷积层采用Conv2d->ReLU->BatchNorm2d结构，测试准确率可以达到85%
#15epoch,CrossEntropyloss=0.021,卷积层采用Conv2d->BatchNorm2d->ReLU结构，测试准确率可以达到87%
import torch.nn as nn

class CifarNet(nn.Module):
    def __init__(self,**kwargs):
        super(CifarNet,self).__init__(**kwargs)
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64,64,3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128,128,3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128,256,3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(256,256,3,stride=1,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.maxpool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AvgPool2d(2,2)
        self.globalavgpool = nn.AvgPool2d(8,8)
        self.drop10 = nn.Dropout(0.1)
        self.drop50 = nn.Dropout(0.5)
        self.fc = nn.Linear(256,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.drop10(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = self.drop10(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.globalavgpool(x)
        x = self.drop50(x)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out
