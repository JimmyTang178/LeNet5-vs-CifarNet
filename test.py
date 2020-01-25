import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
import matplotlib.pyplot as plt
from LeNet5 import LeNet
from model import CifarNet
model_path = './CifarNet_new.pkl'
test_path ='./data/cifar-10-batches-py/test_batch'
batch_size = 1
transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_data = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size= batch_size,
                                 shuffle=True,num_workers=1)
test_data = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True,num_workers=1)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def load_cifar_batch(path):
    with open(path,'rb') as f:
        data = pickle.load(f,encoding="latin1")
        x = data['data'].reshape(10000,3,32,32).transpose(0,2,3,1).astype(np.int32)
        y = np.array(data['labels']).astype(np.int32)
        return x,y
def show_img():
    imgs,labels = load_cifar_batch(test_path)
    i = np.random.randint(0,10000)
    img = np.reshape(imgs[i],[32,32,3])
    #plt.imshow(img)
    #plt.title(classes[labels[i]])
    #print(classes[labels[i]])
    #plt.pause(1)
    return img


if __name__ == '__main__':
    net = CifarNet()
    #net = LeNet()
    net.load_state_dict(torch.load(model_path))
    net = net.eval()##测试阶段若使用了BN且batchsize与训练阶段不同，则需要将模型设置为推理模式model.eval()。此处batchsize=1,训练时batchsize=100
    print(net)
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in test_loader:
            #print(x.shape)
            #img = x.numpy()
            #img = img.squeeze()
            #img = img.transpose(1,2,0).astype(np.float32)
            out = net(x)
            _,predicted = torch.max(out.data,1)#torch.max(tensor,1)返回每一行最大值的数值和对应的索引
            #print("predict:",classes[predicted])
            #print("label  :",classes[y])
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print('Accuracy:%d'%(correct*100 /total)+'%')

