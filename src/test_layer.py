import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,64,drop_last=True)

class Layer(nn.Module):
    def __init__(self):
        super(Layer,self).__init__()
        self.Linear1 = Linear(196608,10)
    def forward(self,input):
        output = self.Linear1(input)
        return output

M = Layer()

step = 0
writer = SummaryWriter("logs")

for data in dataloader:
    imgs,targets = data
    # print(imgs.shape)
    # imgs = torch.reshape(imgs,(1,1,1,-1))
    imgs = torch.flatten(imgs)
    outputs = M(imgs)
    print(imgs.shape)
    print(outputs.shape)