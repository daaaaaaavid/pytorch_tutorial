import torchvision
import time
from torch.utils.tensorboard import SummaryWriter

from model_1 import *
from torch.utils.data import DataLoader

#準備數據集

train_data = torchvision.datasets.CIFAR10("../train_dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10("../test_dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#加載數據集

train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

#data length
train_data_size = len(train_data)
test_data_size = len(test_data)

#build model
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
M = Model()
M = M.cuda()

#lost function

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
#optimizer

learning_rate = 1e-2
optim = torch.optim.SGD(M.parameters(),lr=learning_rate)

#set parameters

total_train_step = 0
total_test_step = 0
epoch = 10

#tensorboard
writer = SummaryWriter("../logs")
start_time = time.time()

for i in range(epoch):
    print("--------- training round {} start ----------".format(i+1))
    for data in train_loader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()

        output = M(imgs)
        loss = loss_fn(output,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        if total_train_step%100 == 0:
            end_time = time.time()
            print("time: {:.3} sec, trainning times: {} , loss: {}".format(end_time-start_time,total_train_step,loss.item()))
            writer.add_scalar("train loss",loss.item(),total_train_step)
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = M(imgs)
            loss = loss_fn(output,targets)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1)==targets).sum()
            total_accuracy += accuracy

    print("total test loss: {}".format(total_test_loss))
    print("total test accuracy: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("total test loss",total_test_loss,total_test_step)
    writer.add_scalar("accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step += 1