import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def load_data(csv_file):
    df = pd.read_csv(csv_file)

    # Split train and test data
    train_data, test_data = train_test_split(df, test_size=0.3)

    # Get label and image, and reshape image
    train_label = train_data['label'].values
    train_image = train_data.iloc[:, 1:].as_matrix() / 255
    train_image = train_image.astype('float').reshape(-1, 1, 28, 28)

    test_label = test_data['label'].values
    test_image = test_data.iloc[:, 1:].as_matrix() / 255
    test_image = test_image.astype('float').reshape(-1, 1, 28, 28)

    # Convert to torch tensor
    train_image = torch.FloatTensor(train_image)
    test_image = torch.FloatTensor(test_image)

    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)

    return (train_image, train_label), (test_image, test_label)

def get_dataloader(data, batch_size=16):
    dataset = TensorDataset(*data)
    loader = DataLoader(dataset, batch_size)
    return loader
