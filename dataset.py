import numpy as np
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from skimage.color import lab2rgb, rgb2lab, rgb2gray

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


class GrayscaleImageFolder(datasets.ImageFolder):
  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img_original = self.transform(img)
      img_original = np.asarray(img_original)
      img_lab = rgb2lab(img_original)
      img_lab = (img_lab + 128) / 255
      img_ab = img_lab[:, :, 1:3]
      img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
      img_original = rgb2gray(img_original)
      img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img_original, img_ab


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

