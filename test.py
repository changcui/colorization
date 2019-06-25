import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from dataset import ColorDataset
from model import ColorNet


test_set = ColorDataset('test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True,
                                num_workers=4)
model = ColorNet()
model.load_state_dict(torch.load('Weights/weights.pkl'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def test():
    model.eval()
    print("Test: Begin!")
    for idx, (data, label) in enumerate(test_loader):
        l, ab = data.to(device), label.to(device)
        for img in l:
            gray_img = np.zeros((224, 224, 3))
            gray_img[:, :, 0] = img.cpu().squeeze().numpy() * 100
            gray_img = gray_img.astype(np.float64)
            gray_img = lab2rgb(gray_img)
            plt.imsave('./Output/' + str(idx) + '_gray.jpg', gray_img)

        output = model(l)
        color_img = torch.cat((l, output), 1)
        gt_img = torch.cat((l, ab), 1)
        color_img = color_img.data.cpu().numpy().transpose((0, 2, 3, 1))
        gt_img = gt_img.data.cpu().numpy().transpose((0, 2, 3, 1))

        for img in color_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            plt.imsave('./Output/' + str(idx) + '_color.jpg', img)
            mse = torch.pow(ab - output, 2).sum()
            print("The MSE of the {:d} image is {:f}".format(idx, mse.item()))
            
        for img in gt_img:
            img[:, :, 0:1] = img[:, :, 0:1] * 100
            img[:, :, 1:3] = img[:, :, 1:3] * 255 - 128
            img = img.astype(np.float64)
            img = lab2rgb(img)
            plt.imsave('./Output/' + str(idx) + '_gt.jpg', img)
        if idx == 20:
            break


if __name__ == "__main__":
    test()
