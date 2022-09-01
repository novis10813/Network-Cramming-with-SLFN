import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    
    def __init__(self, data, device) -> None:
        
        # load csv data
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        
        # feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        
        
        # convert to tensors
        self.num_class = len(set(y))
        
        # binary and real number
        if self.num_class <= 2:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)
            self.y = torch.tensor(y, dtype=torch.float32, device=device).view(-1)
            # self.y = torch.tensor(y, dtype=torch.float32, device=device)
            
        # multi
        elif self.num_class > 2:
            self.X = torch.tensor(X, dtype=torch.float32, device=device)
            self.y = torch.tensor(y, dtype=torch.long, device=device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index].view(-1)


def create_dataloader(datapath='train_all_0.csv', batch_size=128, random_state=87, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    # read the data
    data = pd.read_csv(datapath, header=None)
    
    # split the data and create dataset
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=random_state)
    train_dataset = CustomDataset(train_data, device)
    valid_dataset = CustomDataset(valid_data, device)

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def LTS_dataloader(dataset, index, batch_size):
    new_dataset = Subset(dataset, index)
    new_dataloader = DataLoader(new_dataset, batch_size)
    
    return new_dataloader