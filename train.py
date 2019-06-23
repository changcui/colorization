import os
import torch.optim as optim
import torch.nn.functional as F
from model import ColorNet
from dataset import ColorDataset 
from common import Config

cfg = Config()
train_set = ColorDataset('train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, 
                shuffle=True, num_workers=4)

model = ColorNet()
if os.path.exists('weights/weights.pkl'):
    model.load_state_dict(torch.load('weights/weights.pkl'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adadelta(model.parameters())

def train():
    model.train()
    
    for epoch in range(cfg.epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            l, ab = data.to(device), label.to(device)
            output = model(l)
            loss = torch.pow((output, ab), 2).sum() / output.numel()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * data.size(0), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                torch.save(model.state_dict(), 'weights/weights.pkl')
    torch.save(model.state_dict(), 'weights/weights.pkl')

if __name__ == '__main__':
    train()
