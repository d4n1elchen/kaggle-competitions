import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, channels, kernel_sizes, dense_layers, n_classes, img_size, batch_norm=True, dropout_p=None):
        super(Cnn, self).__init__()

        self.n_classes = n_classes
        self.img_size = img_size
        self.conv_img_size = self.img_size # Assume img is square, so only one dimension
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p

        self.conv_layer_size = len(channels)
        self.dense_layer_size = len(dense_layers)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * self.conv_layer_size

        self.conv = []
        self.conv_bn = []

        self.conv.append(nn.Conv2d(1, channels[0], kernel_sizes[0]))
        self.add_module('Conv0', self.conv[0])

        if batch_norm:
            self.conv_bn.append(nn.BatchNorm2d(channels[0]))
            self.add_module('Conv_bn0', self.conv_bn[0])

        self.conv_img_size = math.floor((self.conv_img_size - (kernel_sizes[0]-1))/2)

        for i in range(1, self.conv_layer_size):
            self.conv.append(nn.Conv2d(channels[i-1], channels[i], kernel_sizes[i]))
            self.add_module('Conv'+str(i), self.conv[i])

            if batch_norm:
                self.conv_bn.append(nn.BatchNorm2d(channels[i]))
                self.add_module('Conv_bn'+str(i), self.conv_bn[i])

            self.conv_img_size = math.floor((self.conv_img_size - (kernel_sizes[i]-1))/2)

        self.pool = nn.MaxPool2d(2, 2)

        if dropout_p:
            self.dropout = nn.Dropout(dropout_p)

        self.conv_flat_size = channels[-1] * self.conv_img_size * self.conv_img_size

        self.dense = []
        self.dense.append(nn.Linear(self.conv_flat_size, dense_layers[0]))
        self.add_module('Dense0', self.dense[0])
        for i in range(1, self.dense_layer_size):
            self.dense.append(nn.Linear(dense_layers[i-1], dense_layers[i]))
            self.add_module('Dense'+str(i), self.dense[i])

        self.output_layer = nn.Linear(dense_layers[-1], n_classes)

    def forward(self, x):
        for i in range(self.conv_layer_size):
            x = self.conv[i](x)
            if self.batch_norm:
                x = self.conv_bn[i](x)
            x = self.pool(F.relu(x))

        x = self.dropout(x)

        x = x.view(-1, self.conv_flat_size)

        for i in range(self.dense_layer_size):
            x = F.relu(self.dense[i](x))
            if self.dropout_p:
                x = self.dropout(x)

        x = self.output_layer(x)

        return x
