import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()
from networks import ResNextEmbeddingNet, SiameseNet
from datasets import SiameseXRayParcels
from losses import ContrastiveLoss
from metrics import AccumulatedAccuracyMetric

siamese_train_dataset = SiameseXRayParcels('train.csv', train=True, transform=True)
siamese_test_dataset = SiameseXRayParcels('test.csv', train=False, transform=False)
batch_size = 1
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = ResNextEmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()

loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, [AccumulatedAccuracyMetric()])
