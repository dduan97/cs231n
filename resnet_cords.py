from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, OLRandomDataLoader, \
    CRAIGDataLoader, GradMatchDataLoader, RandomDataLoader
from dotmap import DotMap
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

import numpy as np

import torchvision.models as models
from tqdm.notebook import tqdm
import composer.functional as cf
import time

num_epochs = 5

NUM_TRAIN = 49000


#TODO: remove the CPU copy operations
def train_and_eval(model, train_loader, test_loader):
  torch.manual_seed(42)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = model.to(device)
  opt = torch.optim.Adam(model.parameters())
  for epoch in range(num_epochs):
    print(f'---- Beginning epoch {epoch} ----')
    model.train()
    # progress_bar = tqdm(train_loader)
    for X, y, _ in train_loader:
      # NOTE: uncomment this next line instead of the previous line if not using a CORDS train_loader
      # for X, y in train_loader:
      X = X.to(device)
      y_hat = model(X).to(device)
      y = y.to(device)
      loss = F.cross_entropy(y_hat, y)
      # progress_bar.set_postfix_str( f'train loss: {loss.detach().cpu().numpy():.4f}')
      loss.backward()
      opt.step()
      opt.zero_grad()
    model.eval()
    num_right = 0
    eval_size = 0
    for X, y in test_loader:
      y = y.to(device)
      y_hat = model(X.to(device)).to(device)
      num_right += (y_hat.argmax(dim=1) == y).sum().cpu().numpy()
      eval_size += len(y)
    acc_percent = 100 * num_right / eval_size
    print(f'Epoch {epoch} validation accuracy: {acc_percent:.2f}%')


logger = logging.getLogger('asdf')

datadir = './toy_data'
batch_size = 1024

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_train = dset.CIFAR10(
    datadir, train=True, download=True, transform=transform)
trainloader = DataLoader(
    cifar10_train,
    batch_size=64,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10(
    datadir, train=True, download=True, transform=transform)
valloader = DataLoader(
    cifar10_val,
    batch_size=64,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10(
    datadir, train=False, download=True, transform=transform)
testloader = DataLoader(cifar10_test, batch_size=64)

model = models.resnet18()

# 1m 47s, final validation acc of 67.29% on colab
# 211 seconds on CPU, 66.36% validation acc
loss = nn.CrossEntropyLoss(reduction='none')

dss_args = dict(
    model=model,
    loss=loss,
    eta=0.01,
    num_classes=10,
    num_epochs=300,
    device='cuda',
    fraction=0.1,
    select_every=20,
    kappa=0,
    linear_layer=False,
    selection_type='SL',
    valid=False,
    v1=True,
    lam=0,
    eps=1e-4,
    greedy='Stochastic')
dss_args = DotMap(dss_args)

gradmatch_loader = GradMatchDataLoader(
    trainloader,
    valloader,
    dss_args,
    logger,
    batch_size=20,
    shuffle=True,
    pin_memory=False)

start = time.time()
train_and_eval(model, gradmatch_loader, testloader)
end = time.time()
print('TOTAL training time:', end - start)
