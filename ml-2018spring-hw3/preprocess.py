import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def preprocess(csv_file):
    df = pd.read_csv(csv_file)

    spl = df.pop("feature").str.split(' ', expand=True)
    df = pd.concat([df, spl], axis=1)

    df.to_csv("preprocessed/"+csv_file, index=False)
    return df


if __name__ == "__main__":
    # preprocess("./train.csv")
    preprocess("./test.csv")
