import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
dataset = torchvision.datasets.CIFAR10("datasets",train=False,transform=torchvision.transforms.ToTensor(),
                                        download=False)
dataloader = DataLoader(dataset,batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

M = CNN()
#print(M)

writer = SummaryWriter("logs")
step = 0

for data in dataloader:
    imgs,targets = data
    output = M(imgs)
    writer.add_images("conv1 input",imgs,step)

    #size (64,6,30,30) -> (--,3,30,30)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("conv1 output",output,step)

    step += 1

writer.close()