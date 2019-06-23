import torch
from model import ColorNet
from dataset import ColorDataset
from common import Config


test_set = ColorDataset('test')
test_loader = torch.utils.data.Dataloader(test_set, batch_size=1, shuffle=False, 
                                num_workers=4)
model = model.ColorNet()
model.load_state_dict(torch.load('weights/weights.pkl'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def test():
    
