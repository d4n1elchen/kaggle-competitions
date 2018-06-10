import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, conv_cfg, dense_cfg, n_classes, img_size, batch_norm=True, dropout_p=0.5):
        super(Cnn, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size

        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        # Build conv
        self.conv = self.make_convs(conv_cfg, batch_norm)

        # Calculate flatten nodes size
        self.conv_img_size = img_size / (2 ** conv_cfg.count('M'))
        if self.conv_img_size < 3:
            raise ValueError("Too many pooling.")
        self.conv_flat_size = int(conv_cfg[-2] * self.conv_img_size * self.conv_img_size)

        # Build dense
        self.dense = self.make_denses(dense_cfg, self.conv_flat_size, dropout_p)

        # Output layer
        self.output_layer = nn.Linear(dense_cfg[-1], n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = self.output_layer(x)

        return x

    # From pytorchvision/vgg.py
    def make_convs(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    # From pytorchvision/vgg.py
    def make_denses(self, cfg, in_nodes, dropout_p=0.5):
        layers = []
        for v in cfg:
            linear = nn.Linear(in_nodes, v)
            layers += [linear, nn.ReLU(inplace=True)]
            if dropout_p:
                layers += [nn.Dropout(dropout_p)]
            in_nodes = v
        return nn.Sequential(*layers)

class GrayVgg11BN(nn.Module):
    def __init__(self, n_classes):
        super(GrayVgg11BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
