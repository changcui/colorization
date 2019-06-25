class Config(object):
    def __init__(self):
        self.epochs = 20
        self.batch_size = 128
        self.save_period = 100

if __name__ == '__main__':
    cfg = Config()
    print(cfg.epochs)
    print(cfg.batch_size)

