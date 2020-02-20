# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2019, Adam Bielski
# Copyright (c) 2020, Martynas Janonis

import numpy as np
from torch.nn.modules import distance


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return "Accuracy"


class AccumulatedDistanceAccuracyMetric(Metric):
    """
    If the distance between the two outputs is less than the margin,
    classify as positive; else negative
    """

    def __init__(self, margin):
        self.correct = 0
        self.total = 0
        self.margin = margin

    def __call__(self, outputs, target, loss):
        pred = distance.PairwiseDistance().forward(outputs[0], outputs[1])
        pred = pred.flatten() < self.margin
        self.correct += sum(pred == target[0])
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return "Accuracy"


class TripletAccumulatedDistanceAccuracyMetric(Metric):
    """
    If the distance between the anchor and the positive is less than the distance between the anchor and the negative,
    classify as positive; else negative
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        dist_pos = distance.PairwiseDistance().forward(outputs[0], outputs[1])
        dist_neg = distance.PairwiseDistance().forward(outputs[0], outputs[2])
        pred = dist_pos.flatten() < dist_neg.flatten()
        self.correct += sum(pred == True)
        self.total += pred.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return "Accuracy"
