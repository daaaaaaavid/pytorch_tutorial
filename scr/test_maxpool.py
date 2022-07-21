import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)

input = torch.reshape(input,(-1,1,5,5))

dataset = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,64)

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool,self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool(input)
        return output

M = MaxPool()
output = M(input)
#print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,traget = data
    output = M(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step += 1

writer.close()