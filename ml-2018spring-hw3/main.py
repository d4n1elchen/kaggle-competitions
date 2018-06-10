import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from Net import Cnn, GrayVgg11BN
from Trainer import Trainer
from Dataset import load_train, load_challenge, get_dataloader
from Visualization import imshow

TRAIN_FILE_PATH = "./preprocessed/train.csv"

EPOCH = 500
BATCH_SIZE = 256
SAVE_MODEL = 'cnn1.pth'

# VGG configs
# Notice that these config is one pooling layer less then original one
#   since the image size is small
cfg = {
    'sVGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'sVGG11': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'sVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'sVGG15': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
}

# Load data
train, test = load_train(TRAIN_FILE_PATH)
train_loader = get_dataloader(train, batch_size=BATCH_SIZE, augment=True)
test_loader = get_dataloader(test, batch_size=BATCH_SIZE, augment=False)

# Show first batch
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# imshow(images, labels)

# Init network and trainer
cnn = Cnn(
    conv_cfg=cfg['sVGG13'],
    dense_cfg=[4096, 4096],
    n_classes=7,
    img_size=48,
    batch_norm=True,
    dropout_p=0.5)
# cnn = GrayVgg11BN(7)
trainer = Trainer(cnn,
                  criterion=nn.CrossEntropyLoss,
                  optimizer=optim.SGD,
                  optim_params={"lr": 0.01,
                                "momentum": 0.9,
                                "nesterov": True})
trainer.load_model(SAVE_MODEL)

# Print network structure
print(cnn)

# Training
print('Start training, training data size = {}'.format(len(train[0])))
trainer.train(train_loader, epoch=EPOCH)
trainer.save_model(SAVE_MODEL)
print('Finished training\n')

# Testing
print('Start testing, testing data size = {}'.format(len(test[0])))
trainer.test(test_loader)
print('Finished testing\n')
