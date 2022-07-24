import torchvision
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

M = Model()

#lost function

loss_fn = nn.CrossEntropyLoss()

#optimizer

learning_rate = 1e-2
optim = torch.optim.SGD(M.parameters(),lr=learning_rate)

#set parameters

total_train_step = 0
total_test_step = 0
epoch = 10

#tensorboard
writer = SummaryWriter("../logs")

for i in range(epoch):
    print("--------- training round {} start ----------".format(i+1))
    for data in train_loader:
        imgs,targets = data
        output = M(imgs)
        loss = loss_fn(output,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        if total_train_step%100 == 0:
            print("trainning times: {} , loss: {}".format(total_train_step,loss.item()))
            writer.add_scalar("train loss",loss.item(),total_train_step)
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
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