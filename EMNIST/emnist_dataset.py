from typing import Tuple, Union
import numpy as np
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from torchvision import datasets
from nn_lib.data import Dataset
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from EMNIST.mlp_classifier import MLPClassifier

train_data = datasets.EMNIST('C:/Users/mosto/PycharmProjects/data', train=True, download=False, split='digits', transform=transforms.Normalize((0.1307,), (0.3081,)))


class EMnistDataset(Dataset):

    def __init__(self, n_samples: int, seed: int = 0, **kwargs):
        random.seed(seed)
        r=random.randint(0, len(train_data.data))
        self.n_samples = n_samples
        self.data = (train_data.data[r:r+n_samples].view(n_samples, 784)/255)
        self.labels = F.one_hot(train_data.targets[r:r+n_samples], num_classes=10).to(dtype=torch.float)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        return self.n_samples


if __name__ == '__main__':
    dataset = EMnistDataset(1000, 21)
    print(dataset[0])

