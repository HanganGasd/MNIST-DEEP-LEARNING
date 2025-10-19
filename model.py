from turtle import forward
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(777)
torch.manual_seed(777)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(777)

training_epochs = 15
batch_size = 128

mnist_train = dsets.MNIST(root='MNIST_data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = dsets.MNIST(root='MNIST_data/',train=False,transform=transforms.ToTensor(),download=True)
data_loader = DataLoader(dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True), 
                           batch_size=128, shuffle=True, drop_last=True)

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride = 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1,padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4*4*128,625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(625,10, bias = True)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

total_batch = len(data_loader)
model.train()

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        hypothesis = model(X)
        cost = criterion(hypothesis, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    model.eval()   

    X_test = mnist_test.data.view(mnist_test.data.shape[0], 1, 28, 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())


torch.save(model.state_dict(), 'mnist_cnn.pth')
print("모델 저장됨: mnist_cnn.pth")


