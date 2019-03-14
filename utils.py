
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K



def get_optimizer(opt):
    if opt.optimizer == 'SGD':
        optimizer = SGD(lr=opt.lr, momentum=opt.momentum, decay=opt.weight_decay)
    elif opt.optimizer == 'Adam':
        optimizer = Adam(lr=opt.lr, decay=opt.weight_decay)

    return optimizer


class SGDRScheduler_with_WarmUp(Callback):

    def __init__(self, min_lr, max_lr, steps_per_epoch, lr_decay=1, cycle_length=10, multi_factor=2, warm_up_epoch=5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay = lr_decay
        self.cycle_length = cycle_length
        self.multi_factor = multi_factor
        self.warm_up_epoch = warm_up_epoch

        self.is_warming = True

        self.history = {}

    def sgdr_lr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        print('', fraction_to_restart)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def warm_lr(self):
        lr = self.max_lr * (self.warm_up_batch / (self.steps_per_epoch * self.warm_up_epoch)) * (self.warm_up_batch / (self.steps_per_epoch * self.warm_up_epoch))
        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.warm_up_batch = 1
        K.set_value(self.model.optimizer.lr, self.warm_lr())

    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k ,[]).append(v)
        
        if self.is_warming:
            self.warm_up_batch += 1
            K.set_value(self.model.optimizer.lr, self.warm_lr())
        else:
            self.batch_since_restart += 1
            K.set_value(self.model.optimizer.lr, self.sgdr_lr())

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == self.warm_up_epoch:
            self.is_warming = False
            self.batch_since_restart = 0
            self.next_restart = self.cycle_length + epoch
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch >= self.warm_up_epoch:
            if epoch + 1 == self.next_restart:
                self.batch_since_restart = 0
                self.cycle_length = np.ceil(self.cycle_length * self.multi_factor)
                self.next_restart += self.cycle_length
                self.max_lr *= self.lr_decay
                self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)

class PrintLearningRate(Callback):
    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        if batch > 0:
            print(' - lr: %.6f'%K.get_value(self.model.optimizer.lr))

class TrainPrint(Callback):
    def __init__(self, steps_per_epoch, max_epoch):
        self.steps_per_epoch = steps_per_epoch
        self.max_epoch = max_epoch
        self.log = 'epoch [%.3d]/[%.3d] batch [%d/%d] loss %.4f lr %.6f acc %.2f'

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = int(epoch)

    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        # loss = float(logs['loss'])
        # lr = float(K.get_value(se))
        print(self.log%(self.epoch, self.max_epoch, batch, self.steps_per_epoch, logs['loss'], K.get_value(self.model.optimizer.lr), logs['acc']))

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if 'val_loss' and 'val_acc' in logs.keys():
            print('Validate on epoch {} : loss {} acc {}'.format(epoch, logs['val_loss'], logs['val_acc']))

