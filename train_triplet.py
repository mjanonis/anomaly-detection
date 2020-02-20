# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2019, Adam Bielski
# Copyright (c) 2020, Martynas Janonis

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np

cuda = torch.cuda.is_available()
from networks import ResNextEmbeddingNet, TripletNet
from datasets import TripletXRayParcels
from losses import TripletLoss
from metrics import TripletAccumulatedDistanceAccuracyMetric

triplet_train_dataset = TripletXRayParcels(
    "triplet_train.csv", train=True, transform=True
)
triplet_test_dataset = TripletXRayParcels(
    "triplet_test.csv", train=False, transform=False
)
batch_size = 1
kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(
    triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
)
triplet_test_loader = torch.utils.data.DataLoader(
    triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs
)

margin = 2.0
embedding_net = ResNextEmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()

loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.99, last_epoch=-1)
n_epochs = 50
log_interval = 10

fit(
    triplet_train_loader,
    triplet_test_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    [TripletAccumulatedDistanceAccuracyMetric()],
)
