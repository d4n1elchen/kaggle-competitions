import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, channels, kernel_sizes, dense_layers, n_classes, img_size):
        super(Cnn, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.conv_img_size = self.img_size # Assume img is square, so only one dimension

        self.conv_layer_size = len(channels)
        self.dense_layer_size = len(dense_layers)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * self.conv_layer_size

        self.conv = []
        self.conv.append(nn.Conv2d(1, channels[0], kernel_sizes[0]))
        self.conv_img_size = math.floor((self.conv_img_size - (kernel_sizes[0]-1))/2)
        for i in range(1, self.conv_layer_size):
            self.conv.append(nn.Conv2d(channels[i-1], channels[i], kernel_sizes[i]))
            self.conv_img_size = math.floor((self.conv_img_size - (kernel_sizes[i]-1))/2)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_flat_size = channels[-1] * self.conv_img_size * self.conv_img_size

        self.dense = []
        self.dense.append(nn.Linear(self.conv_flat_size, dense_layers[0]))
        for i in range(1, self.dense_layer_size):
            self.dense.append(nn.Linear(dense_layers[i-1], dense_layers[i]))

        self.output_layer = nn.Linear(dense_layers[-1], n_classes)

    def forward(self, x):
        for i in range(self.conv_layer_size):
            x = self.pool(F.relu(self.conv[i](x)))

        x = x.view(-1, self.conv_flat_size)

        for i in range(self.dense_layer_size):
            x = F.relu(self.dense[i](x))

        x = self.output_layer(x)

        return x
