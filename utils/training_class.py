from utils.training_utils import save_model_at_fold


class EarlyStopping:
    def __init__(self, epoch_interval=5):
        self.best_metric = 0
        self.best_epoch = 0
        self.epoch_interval = epoch_interval
        self.early_stop = False

    def __call__(self, metric, epoch):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
        if epoch >= self.best_epoch + self.epoch_interval:
            self.early_stop = True


class SaveModel:
    def __init__(self, epoch_interval=5):
        self.models = {}
        self.epoch_interval = epoch_interval
        self.epoch = 0

    def __call__(self, new_model, epoch):
        self.models[epoch] = new_model
        self.epoch = epoch
        if len(self.models) > self.epoch_interval:
            del self.models[epoch - self.epoch_interval]

    def save_best_model(self, args, if_stop):
        if if_stop:
            save_model_at_fold(self.models[0], args)
            print(f'Model training stops at {self.epoch}')
            print(f'Model saved at {self.epoch-self.epoch_interval}')


if __name__ == "__main__":
    a = SaveModel()
    for i in range(10):
        model_sample = i+0.1*i
        a(model_sample, i)
    for key1 in a.models:
        print(key1, a.models[key1])


