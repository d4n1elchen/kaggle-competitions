import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from Net import Cnn
from Dataset import load_data, get_dataloader
from Visualization import imshow

DATA_FILE_PATH = "./train.csv"
train, test = load_data(DATA_FILE_PATH)

train_loader = get_dataloader(train)
test_loader = get_dataloader(test)

dataiter = iter(train_loader)
images, labels = dataiter.next()

# imshow(images, labels)

cnn = Cnn(
    channels=(32, 64),
    kernel_sizes=5,
    dense_layers=(1024,),
    n_classes=10,
    img_size=28)

if torch.cuda.is_available():
    cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters())

print('Start training, training data size = {}'.format(len(train[0])))

for epoch in range(10):

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
            print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = cnn(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the {} test images: {}%'.format(total, 100 * correct / total))
