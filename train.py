import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import ColorNet
from dataset import ColorDataset 
from common import Config, AverageMeter
from tqdm import tqdm

cfg = Config()
train_set = ColorDataset('train')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, 
                shuffle=True, num_workers=4)

model = ColorNet()
if os.path.exists('Weights/weights.pkl'):
    model.load_state_dict(torch.load('Weights/weights.pkl'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

def train():
    model.train()
    print("Train: Begin!")  
    losses = AverageMeter()
    for epoch in range(cfg.epochs):
        for batch_idx, (data, label) in enumerate(tqdm((train_loader))):
            l, ab = data.to(device), label.to(device)
            output = model(l)
            loss = criterion(output, ab)
            losses.update(loss.item(), l.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % cfg.save_period == 0:
                print('Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * data.size(0), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), losses.avg))
                torch.save(model.state_dict(), 'Weights/weights.pkl')
    print("Saving the model!")
    torch.save(model.state_dict(), 'Weights/weights.pkl')
    print("Done")

if __name__ == '__main__':
    train()
