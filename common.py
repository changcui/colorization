class Config(object):
    def __init__(self, phrase):
        self.epoches = 20
        self.batch_size = 32
        self.save_period = 100

if __name__ == '__main__':
    cfg = Config()
    print(cfg.epoches)
    print(cfg.batch_size)

