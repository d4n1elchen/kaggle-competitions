import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils

def load_train(csv_file):
    df = pd.read_csv(csv_file)

    # Split train and test data
    train_data, test_data = train_test_split(df, test_size=0.05)

    # Get label and image, and reshape image
    train_label = train_data['label'].values
    train_image = train_data.iloc[:, 1:].as_matrix() / 255
    train_image = train_image.astype('float').reshape(-1, 1, 48, 48)

    test_label = test_data['label'].values
    test_image = test_data.iloc[:, 1:].as_matrix() / 255
    test_image = test_image.astype('float').reshape(-1, 1, 48, 48)

    # Convert to torch tensor
    train_image = torch.FloatTensor(train_image)
    test_image = torch.FloatTensor(test_image)

    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)

    return (train_image, train_label), (test_image, test_label)

def load_challenge(csv_file):
    df = pd.read_csv(csv_file)

    image = df.iloc[:, 1:].as_matrix() / 255
    image = image.reshape(-1, 1, 48, 48)

    # Convert to torch tensor
    image = torch.FloatTensor(image)

    return (image,)

def get_dataloader(data, batch_size=16, augment=True):
    if len(data) == 2:
        dataset = TrainTensorDataset(data, augment)
    elif len(data) == 1:
        dataset = ChallengeTensorDataset(data[0])

    loader = DataLoader(dataset, batch_size)
    return loader

class TrainTensorDataset(TensorDataset):
    def __init__(self, tensors, augment=True):
        super(TrainTensorDataset, self).__init__(*tensors)
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomCrop(48, padding=8),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image, label = super(TrainTensorDataset, self).__getitem__(index)
        if self.augment:
            image = self.transform(image)
        return (image, label)

class ChallengeTensorDataset(TensorDataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

