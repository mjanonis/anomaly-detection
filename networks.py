# SPDX-License-Identifier: EUPL-1.2

# Unmodified code written by Adam Bielski is licensed under the BSD-3-Clause license

# All further additions and modifications: Copyright (c) 2020, Martynas Janonis

# Licensed under the EUPL-1.2-or-later

import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.convnet = network

        # Strip the classification layer
        self.convnet = nn.Sequential(*(list(self.convnet.children())[:-1]))
        # Add an avg pooling layer (need output shape of [?, 1024, 1, 1])
        self.aavgp3d = nn.AdaptiveAvgPool3d(output_size=(1024, 1, 1))
        self.flat = nn.Flatten()

    def forward(self, x):
        output = self.convnet(x)
        output = self.aavgp3d(output)
        output = self.flat(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
