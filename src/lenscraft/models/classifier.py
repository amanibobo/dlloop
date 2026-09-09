"""Supervised classifiers from Alexander et al. 2019 (DESIGN.md Section 7.1).

Standard torchvision ResNet-18 / AlexNet, untrained, with a single-channel stem and a 3-class
head (none / subhalo / vortex). Both accept any input size >= ~64 px thanks to adaptive pooling.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def build_classifier(arch: str = "resnet18", n_classes: int = 3) -> nn.Module:
    if arch == "resnet18":
        model = tvm.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # grayscale input
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "alexnet":
        model = tvm.alexnet(weights=None)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    else:
        raise ValueError(f"unknown classifier arch {arch!r}; expected 'resnet18' or 'alexnet'")
    return model
