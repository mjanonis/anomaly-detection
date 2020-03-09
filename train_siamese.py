# SPDX-License-Identifier: EUPL-1.2

# Unmodified code written by Adam Bielski is licensed under the BSD-3-Clause license

# All further additions and modifications: Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision.models import resnext101_32x8d, densenet201

from trainer import fit

cuda = torch.cuda.is_available()
from networks import EmbeddingNet, SiameseNet
from datasets import SiameseXRayParcels
from losses import ContrastiveLoss
from metrics import AccumulatedDistanceAccuracyMetric

siamese_train_dataset = SiameseXRayParcels("train.csv", train=True, transform=True)
siamese_test_dataset = SiameseXRayParcels("test.csv", train=False, transform=False)
batch_size = 1
kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(
    siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
)
siamese_test_loader = torch.utils.data.DataLoader(
    siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs
)

margin = 2.0
embedding_net = EmbeddingNet(resnext101_32x8d(pretrained=True))
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()

loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.99, last_epoch=-1)
n_epochs = 50
log_interval = 10

fit(
    siamese_train_loader,
    siamese_test_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    [AccumulatedDistanceAccuracyMetric(margin // 2)],
)
