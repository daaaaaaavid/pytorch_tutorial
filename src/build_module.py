from torch import nn


class module(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input):
        return input+1

M = module()
print(M.forward(1))