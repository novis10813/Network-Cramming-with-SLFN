import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    
    def __init__(self, filepath, device) -> None:
        
        # load csv data
        data = pd.read_csv(filepath, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        
        # convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloader(datapath='train_all_0.csv', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    dataset = CustomDataset(datapath, device)

    # create data indices for train val split
    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(np.floor(0.2 * data_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # create data loader
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)
    
    return train_loader, val_loader