import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import lab2rgb
from dataset import GrayscaleImageFolder
from model import ColorNet
from common import Config


cfg = Config()
val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
val_imagefolder = GrayscaleImageFolder('images/val' , val_transforms)
val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=1, shuffle=False)

model = ColorNet()
model.load_state_dict(torch.load('Weights/weights.pkl'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def test():
    model.eval()
    print("Test: Begin!")
    for idx, (data, label) in enumerate(val_loader):
        if idx % 20:
            continue
        l, ab = data.to(device), label.to(device)
        for img in l:
            gray_img = np.zeros((224, 224, 3))
            gray_img[:, :, 0] = img.cpu().squeeze().numpy() * 100
            gray_img = gray_img.astype(np.float64)
            gray_img = lab2rgb(gray_img)
            plt.imsave('./Output/' + str(idx) + '_gray.jpg', gray_img)

        output = model(l)
        if cfg.is_classification:
            color_img = torch.cat((l, output[0], output[1]), 1)
        else:
            color_img = torch.cat((l, output), 1)
        gt_img = torch.cat((l, ab), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        gt_img = gt_img.data.cpu().numpy().transpose((0, 2, 3, 1))

        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            from IPython import embed
            embed()
            if cfg.is_classification:
                a_ = img[:, :, 1:26].argmax(axis=2) * 1.0 / cfg.bins
                b_ = img[:, :, 26:51].argmax(axis=2) * 1.0 / cfg.bins
                img[:, :, 1] = a_.data
                img[:, :, 2] = b_.data
                img = img[:, :, 0:3]
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            plt.imsave('./Output/' + str(idx) + '_color.jpg', img)
            
        for img in gt_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            plt.imsave('./Output/' + str(idx) + '_gt.jpg', img)


if __name__ == "__main__":
    test()
