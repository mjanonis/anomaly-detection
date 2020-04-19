# SPDX-License-Identifier: EUPL-1.2
# Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import torch
import sys

from torchvision.models import resnext101_32x8d, densenet201
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, cohen_kappa_score
from networks import EmbeddingNet, TripletNet
from datasets import XRayParcels
from joblib import dump, load

cuda = torch.cuda.is_available()
device = torch.device("cuda") if cuda else torch.device("cpu")

xray_train_dataset = XRayParcels("svm_train.csv", train=True, transform=True)
xray_test_dataset = XRayParcels("svm_test.csv", train=False, transform=False)
batch_size = 16
kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
xray_train_loader = torch.utils.data.DataLoader(
    xray_train_dataset, batch_size=batch_size, shuffle=True, **kwargs
)
xray_test_loader = torch.utils.data.DataLoader(
    xray_test_dataset, batch_size=batch_size, shuffle=False, **kwargs
)

# Initialize and load the embedding network
embedding_net = EmbeddingNet(densenet201())
model = TripletNet(embedding_net)
model.load_state_dict(torch.load("triplet_densenet201_m2.pth", map_location=device))
model.eval()

# Initialize the SVM
svm = SGDClassifier(loss="hinge", verbose=0, class_weight={0: 1, 1: 5}, warm_start=True)

n_epochs = 1
highest_kappa = 0

for epoch in range(n_epochs):

    print("Starting epoch {}".format(epoch))
    # Train stage
    # Generate #batch_size vectors to pass as a dataset to the SVM
    for batch_idx, (data, target) in enumerate(xray_train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        with torch.no_grad():
            vectors = model.embedding_net(*data)

        # Convert from PyTorch tensors to NumPy arrays
        vectors = vectors.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # Do one epoch of SGD for the SVM
        svm.partial_fit(vectors, target, classes=[0, 1])

        message = "Train: [{}/{} ({:.0f}%)]".format(
            batch_idx * len(data[0]),
            len(xray_train_loader.dataset),
            100.0 * batch_idx / len(xray_train_loader),
        )

        sys.stdout.write("\x1b[2K")  # Clear to the end of line
        sys.stdout.write("\r" + message)
        sys.stdout.flush()

    print("Starting validation")
    # Test stage
    # Generate #batch_size vectors to pass as a dataset to the SVM
    y_pred = []
    y_true = []
    for batch_idx, (data, target) in enumerate(xray_test_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
           
        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        with torch.no_grad():
            vectors = model.embedding_net(*data)

        # Convert from PyTorch tensors to NumPy arrays
        vectors = vectors.detach().cpu().numpy()
        y_true += target.detach().cpu().numpy()

        y_pred += svm.predict(vectors)

    print(
        "Epoch {}/{}. Validation set: Avg. accuracy: {:.4f}, avg. F1 score: {:.4f}, avg. AUC: {:.4f}, avg. Kappa {:.4f}".format(
            epoch,
            n_epochs,
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            roc_auc_score(y_true, y_pred),
            cohen_kappa_score(y_true, y_pred),
        )
    )

    # Save the model if Kappa is larger
    if cohen_kappa_score(y_true, y_pred) > highest_kappa:
        dump(svm, "svm.joblib")
