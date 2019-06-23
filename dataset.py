import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class ColorDataset(Dataset):
    def __init__(self, phrase):
        assert (phrase in ['train', 'test'])
        self.transforms = transforms.Compose([transforms.ToTensor()])
        if phrase == 'train':
            self.data = np.load('Dataset/train.npy')
        else:
            self.data = np.load('Dataset/test.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.transforms(self.data[idx])
        return (data_[0, :, :].unsqueeze(0), data_[1:, :, :])


if __name__ == '__main__':
    train_set = ColorDataset('train')
    test_set = ColorDataset('test')
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                                num_workers=4)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False,
                                num_workers=4)

    for data, label in test_loader:
        print(data.shape, label.shape)
        print(data[0], label[0])
        break

