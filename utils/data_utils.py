import torch 
from torch.utils.data import random_split, TensorDataset, DataLoader

def load_data(filepath: str) -> TensorDataset:

    tensor_dataset = torch.load(filepath, weights_only= False) 

    return tensor_dataset


def train_test_split(split_ratio, data: TensorDataset) -> (DataLoader, DataLoader): 

   return random_split(data, [split_ratio, 1-split_ratio])


def get_data_loader(dataset: TensorDataset, batch_size = 64, shuffle = True) -> DataLoader: 

    return DataLoader(dataset, batch_size, shuffle) 

