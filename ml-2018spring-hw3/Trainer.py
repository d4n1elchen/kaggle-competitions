import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

class Trainer():
    def __init__(self, net, criterion, optimizer, optim_params={}, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda

        self.loss_hist = []

        if use_cuda:
            net.cuda()

        self.criterion = criterion()
        self.optimizer = optimizer(net.parameters(), **optim_params)
        print(self.optimizer.state_dict())

    def train(self, train_loader, epoch=1):

        for e in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # Get inputs
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Reset the gradient
                self.optimizer.zero_grad()

                # forward
                outputs = self.net(inputs)

                # loss
                loss = self.criterion(outputs, labels)

                # backward
                loss.backward()

                # update weights
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 100 == 99:
                    print('[{:3d}, {:4d}] loss: {}'.format(e + 1, i + 1, running_loss / 100))
                    self.loss_hist.append(running_loss)
                    running_loss = 0.0

    def test(self, test_loader):
        # Set to evaluation mode
        is_train = self.net.training
        self.net.eval()

        # Init statistic variable
        correct = 0
        total = 0

        # Test all data
        for data in test_loader:
            images, labels = data
            outputs = self.net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        # Print result
        print('Accuracy of the network on the {} test images: {}%'.format(total, 100 * correct / total))

        # Reset training mode
        self.net.train(is_train)

    def save_model(self, save_path):
        torch.save(self.net.state_dict(), save_path)
        print("Save checkpoint to {}".format(save_path))

    def load_model(self, load_path):
        self.net.load_state_dict(torch.load(load_path))
        print("Load checkpoint from {}".format(load_path))
