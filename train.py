#本项目由两个模型组成，一个放在model.py，另一个放在LeNet5.py
#主要就是用于比较两个模型用于cifar10分类的效果
import torch
import time
import torchvision
import torchvision.transforms as transforms
#from process import train_loader
from model import CifarNet
from LeNet5 import LeNet
batch_size = 100
transform = transforms.Compose([
                                transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size= batch_size,
                                 shuffle=True,num_workers=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = CifarNet()
#net = LeNet()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
loss = torch.nn.CrossEntropyLoss()
epochs = 15
ls = []
#print(net)
if __name__ =='__main__':
    for epoch in range(epochs):
        start = time.time()
        for x,y in train_loader:
            out = net(x)
            l = loss(out,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        end = time.time()
        print('epoch:',epoch+1,'loss:',l.item(),'time:%d s'%(end-start))
        ls.append(l.item())
        if l.item() <= min(ls):
           torch.save(net.state_dict(),'CifarNet_new.pkl')
           print('the newest model saved!')
        #torch.save(net.state_dict(), 'CifarNet.pkl')