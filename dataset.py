import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ColorDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data = np.load(data_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.transforms(self.data[idx])
        return (data_[:, :, 0], data_[:, :, 1:])


if __name__ == '__main__':
    transforms_train = transforms.Compose([transforms.ToTensor(), 
                        transforms.Normalize((0.41285187, 0.5139937, 0.5287902),
                        (0.28282014, 0.05259202, 0.07479706))])
    transforms_test = transforms.Compose([transforms.ToTensor(), 
                        transforms.Normalize((0.42568654, 0.513196, 0.5310263), 
                        (0.27751327, 0.05601797, 0.08200557))])
    train_set = ColorDataset('train.npy', transforms_train)
    test_set = ColorDataset('test.npy', transforms_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                                num_workers=4)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False,
                                num_workers=4)

    for data, label in train_loader:
        print(data.shape, label.shape)
        print(data[0], label[0])
        break

