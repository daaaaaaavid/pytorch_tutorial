import torch
import torchvision.datasets
from torch import nn, sigmoid
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))

dataset = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,64)

class Relu(nn.Module):
    def __init__(self):
        super(Relu,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
    def forward(self,input):
        output = self.sigmoid1(input)
        return output

M = Relu()
output = M(input)
# print(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs,targets = data
    outputs = M(imgs)
    writer.add_images("input",imgs,step)
    writer.add_images("output",outputs,step)
    step += 1
writer.close()