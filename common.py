import torch


class Config(object):
    def __init__(self):
        self.epochs = 75
        self.batch_size = 64
        self.save_period = 100
        self.bins = 25
        self.is_classification = True

class AverageMeter(object):
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def one_hot(target, bins):
    y = (target / (1. / bins)).floor().long()
    y_onehot = torch.FloatTensor(y.size(0), bins, y.size(2), y.size(3))
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

if __name__ == '__main__':
    cfg = Config()
    print(cfg.epochs)
    print(cfg.batch_size)
    y = torch.FloatTensor(32, 1, 224, 224).uniform_()
    print(y[0, :, 0, 0], y.shape)
    y_onehot = one_hot(y, 25)
    print(y_onehot[0, :, 0, 0], y_onehot.shape)

