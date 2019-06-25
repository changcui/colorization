class Config(object):
    def __init__(self):
        self.epochs = 20
        self.batch_size = 16
        self.save_period = 100


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


if __name__ == '__main__':
    cfg = Config()
    print(cfg.epochs)
    print(cfg.batch_size)

