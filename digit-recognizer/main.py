import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from Net import Cnn
from Dataset import load_train, load_challenge, get_dataloader
from Visualization import imshow

TRAIN_FILE_PATH = "./train.csv"
CHALLENGE_FILE_PATH = "./test.csv"
OUTPUT_PATH = "./submission.csv"
EPOCH = 10

train, test = load_train(TRAIN_FILE_PATH)

train_loader = get_dataloader(train, batch_size=32)
test_loader = get_dataloader(test, batch_size=32)

dataiter = iter(train_loader)
images, labels = dataiter.next()

# imshow(images, labels)

cnn = Cnn(
    channels=(32, 64),
    kernel_sizes=5,
    dense_layers=(1024,),
    n_classes=10,
    img_size=28,
    batch_norm=True,
    dropout_p=0.25)

if torch.cuda.is_available():
    cnn.cuda()

print(cnn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())

print('Start training, training data size = {}'.format(len(train[0])))

for epoch in range(EPOCH):

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Reset the gradient
        optimizer.zero_grad()

        # forward
        outputs = cnn(inputs)

        # loss
        loss = criterion(outputs, labels)

        # backward
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 99:
            print('[{}, {:4d}] loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
cnn.eval()

correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = cnn(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the {} test images: {}%'.format(total, 100 * correct / total))

challenge_images = load_challenge(CHALLENGE_FILE_PATH)
challenge_loader = get_dataloader(challenge_images)

print('Output challenge result')

with open(OUTPUT_PATH, 'w') as f:
    f.write('ImageId,Label\n')
    cnt = 1
    for data in challenge_loader:
        images = data
        output = cnn(Variable(images.cuda()))
        _, predicted = torch.max(output.data, 1)
        for pred in predicted.cpu():
            f.write("{!s},{!s}\n".format(cnt, pred))
            cnt += 1

# dataiter = iter(challenge_loader)
# images = dataiter.next()
# imshow(images)
