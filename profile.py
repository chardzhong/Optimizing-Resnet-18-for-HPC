# -*- coding: utf-8 -*-
"""HPMLHW2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aLz9nXzCYwOJJQ42eAwgsXi8TC8R4NpT

Import
"""

print("Importing")
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score
import argparse
import time

parser = argparse.ArgumentParser(description='Parse those args')
parser.add_argument('--path', type=str, default='./', help='path to store data (default: ./)')
parser.add_argument('--loadworkers', type=int, default=2, help='num_workers for dataloader (default: 2)')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use (default: sgd)')
parser.add_argument('--cuda', action='store_true', default=False, help='enable CUDA training')
parser.add_argument('--removeBN', action='store_true', default=False, help='Remove batch norm layers')
parser.add_argument('--c3', action='store_true', default=False, help='Run C3 experiment')
parser.add_argument('--q3', action='store_true', default=False, help='Run Q3 experiment')

args = parser.parse_args()

datapath = args.path
num_workers = args.loadworkers
opt = args.optimizer
use_cuda = args.cuda
removeBN = args.removeBN
c3 = args.c3
q3 = args.q3

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# train = torchvision.datasets.CIFAR10('./CIFAR10', train=True, download=True, transform=transform)
# test = torchvision.datasets.CIFAR10('./CIFAR10', train=False, download=True, transform=transform)

train = torchvision.datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(datapath, train=False, download=True, transform=transform)

# trainDataLoader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

trainDataLoader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=num_workers)

testDataLoader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)

"""Model"""

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if removeBN: #C7
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                )
            else:
                self.shortcut = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if not removeBN else out #C7
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) if not removeBN else out #C7
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if not removeBN else out #C7
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet18().cuda() if use_cuda else ResNet18() #C5
loss = torch.nn.CrossEntropyLoss()
optimizer = None
match opt: #C6
  case 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  case 'sgdwn':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
  case 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, lr_decay=0, weight_decay=5e-4, initial_accumulator_value=0, eps=1e-10)
  case 'adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1, rho=0.9, eps=1e-06, weight_decay=5e-4)
  case 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

"""C1"""

if not c3:
  print("Training ...")
  train_loss_history = []
  test_loss_history = []
  train_acc_history = []
  loadtimes =[]
  traintimes = []
  totaltimes = []

  for epoch in range(5):
    #epoch time
    if use_cuda: #C2
        torch.cuda.synchronize()
    startepoch = time.perf_counter()

    train_loss = test_loss = 0.0
    predictions = np.empty(0)
    groundtruth = np.empty(0)
    loadtime = 0
    traintime = 0
    totaltime = 0

    # train
    model.train()
    trainiter = iter(trainDataLoader)
    for i in range(len(trainDataLoader)):
      #load time
      if use_cuda: #C2
        torch.cuda.synchronize()
      startload = time.perf_counter()

      images, labels = next(trainiter)

      if use_cuda: #C2
        torch.cuda.synchronize()
      endload = time.perf_counter()
      loadtime += endload - startload

      #Move to device
      images = images.cuda() if use_cuda else images #C5
      labels = labels.cuda() if use_cuda else labels #C5

      #Train time
      if use_cuda: #C2
        torch.cuda.synchronize()
      starttrain = time.perf_counter()

      optimizer.zero_grad()
      outputs = model(images) #forward
      fit = loss(outputs, labels) #calculate loss
      fit.backward() #backprop
      optimizer.step() #update weights
      train_loss += fit.item()

      if use_cuda: #C2
        torch.cuda.synchronize()
      endtrain = time.perf_counter()
      traintime += endtrain - starttrain

      predlabels = torch.max(outputs, dim=1)[1]
      predictions = np.append(predictions, predlabels.cpu().detach().numpy() if use_cuda else predlabels.detach().numpy())
      groundtruth = np.append(groundtruth, labels.cpu().detach().numpy() if use_cuda else labels.detach().numpy())

    # test
    model.eval()
    testiter = iter(testDataLoader)
    for i in range(len(testDataLoader)):
      with torch.no_grad():
        #Load time
        if use_cuda: #C2
          torch.cuda.synchronize()
        startload = time.perf_counter()

        images, labels = next(testiter)

        if use_cuda: #C2
          torch.cuda.synchronize()
        endload = time.perf_counter()
        loadtime += endload - startload

        #Move to device
        images = images.cuda() if use_cuda else images #C5
        labels = labels.cuda() if use_cuda else labels #C5

        outputs = model(images) #forward
        fit = loss(outputs, labels) #calculate loss
        test_loss += fit.item()

    train_loss = train_loss/len(trainDataLoader)
    test_loss = test_loss/len(testDataLoader)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch+1, train_loss, test_loss))

    train_acc_history.append(accuracy_score(predictions, groundtruth))

    #end epoch
    if use_cuda: #C2
        torch.cuda.synchronize()
    endepoch = time.perf_counter()
    totaltime += endepoch - startepoch

    loadtimes.append(loadtime)
    traintimes.append(traintime)
    totaltimes.append(totaltime)

  print('Best Training Accuracy: {:.6f}\n'.format(max(train_acc_history)))

  for i in range(5):
    print('Epoch: {} \tLoad time (sec): {:.6f} \tTrain time (sec): {:.6f} \tTotal time (sec): {:.6f}'.format(i+1, loadtimes[i], traintimes[i], totaltimes[i]))
  print()

"""C3"""

if c3:
  workers = [0,4,8,12]
  totalload = []
  for w in workers:
    print("Num workers: "+str(w))
    trainDataLoader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=w)

    print("Training ...")
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    loadtimes =[]
    traintimes = []
    totaltimes = []

    for epoch in range(5):
      #epoch time
      if use_cuda: #C2
          torch.cuda.synchronize()
      startepoch = time.perf_counter()

      train_loss = test_loss = 0.0
      predictions = np.empty(0)
      groundtruth = np.empty(0)
      loadtime = 0
      traintime = 0
      totaltime = 0

      # train
      model.train()
      trainiter = iter(trainDataLoader)
      for i in range(len(trainDataLoader)):
        #load time
        if use_cuda: #C2
          torch.cuda.synchronize()
        startload = time.perf_counter()

        images, labels = next(trainiter)

        if use_cuda: #C2
          torch.cuda.synchronize()
        endload = time.perf_counter()
        loadtime += endload - startload

        #Move to device
        images = images.cuda() if use_cuda else images #C5
        labels = labels.cuda() if use_cuda else labels #C5

        #Train time
        if use_cuda: #C2
          torch.cuda.synchronize()
        starttrain = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(images) #forward
        fit = loss(outputs, labels) #calculate loss
        fit.backward() #backprop
        optimizer.step() #update weights
        train_loss += fit.item()

        if use_cuda: #C2
          torch.cuda.synchronize()
        endtrain = time.perf_counter()
        traintime += endtrain - starttrain

        predlabels = torch.max(outputs, dim=1)[1]
        predictions = np.append(predictions, predlabels.cpu().detach().numpy() if use_cuda else predlabels.detach().numpy())
        groundtruth = np.append(groundtruth, labels.cpu().detach().numpy() if use_cuda else labels.detach().numpy())

      # test
      model.eval()
      testiter = iter(testDataLoader)
      for i in range(len(testDataLoader)):
        with torch.no_grad():
          #Load time
          if use_cuda: #C2
            torch.cuda.synchronize()
          startload = time.perf_counter()

          images, labels = next(testiter)

          if use_cuda: #C2
            torch.cuda.synchronize()
          endload = time.perf_counter()
          loadtime += endload - startload

          #Move to device
          images = images.cuda() if use_cuda else images #C5
          labels = labels.cuda() if use_cuda else labels #C5

          outputs = model(images) #forward
          fit = loss(outputs, labels) #calculate loss
          test_loss += fit.item()

      train_loss = train_loss/len(trainDataLoader)
      test_loss = test_loss/len(testDataLoader)
      train_loss_history.append(train_loss)
      test_loss_history.append(test_loss)
      print('Epoch: {} \tTraining Loss: {:.6f} \tTesting Loss: {:.6f}'.format(epoch+1, train_loss, test_loss))

      train_acc_history.append(accuracy_score(predictions, groundtruth))

      #end epoch
      if use_cuda: #C2
          torch.cuda.synchronize()
      endepoch = time.perf_counter()
      totaltime += endepoch - startepoch

      loadtimes.append(loadtime)
      traintimes.append(traintime)
      totaltimes.append(totaltime)

    print('Best Training Accuracy: {:.6f}\n'.format(max(train_acc_history)))

    for i in range(5):
      print('Epoch: {} \tLoad time (sec): {:.6f} \tTrain time (sec): {:.6f} \tTotal time (sec): {:.6f}'.format(i+1, loadtimes[i], traintimes[i], totaltimes[i]))
    totalload.append(sum(loadtimes))
    print("Total load time (sec): "+ str(sum(loadtimes))+"\n")

"""Q3"""

if q3:
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Total number of parameters: {total_params}')
  num_gradients = 0
  for name, param in model.named_parameters():
    if param.grad is not None:
        num_gradients += 1
  print(f"Total number of gradients: {num_gradients}")