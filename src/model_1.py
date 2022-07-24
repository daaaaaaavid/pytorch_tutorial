import torch
from torch import nn
from torch.nn import Sequential


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10),
        )
    def forward(self,x):
        output = self.model(x)
        return output

if __name__ == '__main__':
    M = Model()
    input = torch.ones((64,3,32,32))
    output = M(input)
    print(output.size())
