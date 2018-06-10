import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from Net import Cnn, GrayVgg11BN
from Trainer import Trainer
from Dataset import load_train, load_challenge, get_dataloader
from Visualization import imshow

CHALLENGE_FILE_PATH = "./preprocessed/test.csv"
OUTPUT_PATH = "./submission_svgg1.csv"
MODEL_PATH = "cnn1.pth"

BATCH_SIZE = 16

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
challenge_images = load_challenge(CHALLENGE_FILE_PATH)
challenge_loader = get_dataloader(challenge_images, batch_size=BATCH_SIZE)

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
                  optimizer=optim.Adam)
trainer.load_model(MODEL_PATH)

# Test
cnn.eval()
print('Output challenge result to {}'.format(OUTPUT_PATH))
labels = []
with open(OUTPUT_PATH, 'w') as f:
    f.write('id,label\n')
    cnt = 0
    for data in challenge_loader:
        images = data
        output = cnn(Variable(images.cuda()))
        _, predicted = torch.max(output.data, 1)
        for pred in predicted.cpu():
            f.write("{!s},{!s}\n".format(cnt, pred))
            if cnt < BATCH_SIZE:
                labels.append(pred)
            cnt += 1

# Show first batch
dataiter = iter(challenge_loader)
images = dataiter.next()
imshow(images, labels)

