import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
    def forward(self,x):
        output = self.model1(x)
        return output

M = model()
input = torch.ones((64,3,32,32))
print(M(input).shape)
writer = SummaryWriter("../logs")
writer.add_graph(M,input)
writer.close()