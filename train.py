import os
import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from model import nets
from opts import parse_opts
from utils import get_optimizer, SGDRScheduler_with_WarmUp, TrainPrint, PrintLearningRate
from dataset import ucf_dataset


def create_callbacks(opt, steps_per_epoch):
    log_dir = os.path.join(opt.root_path, opt.log_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)

    result_path = os.path.join(opt.root_path, opt.result_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    checkpoint = ModelCheckpoint(os.path.join(result_path, 'ep{epoch:03d}-val_acc{val_acc:.2f}.h5'),
                                 monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
    learning_rate_scheduler = SGDRScheduler_with_WarmUp(0, opt.lr, steps_per_epoch, lr_decay=opt.lr_decay, 
                                                        cycle_length=opt.cycle_length, multi_factor=opt.multi_factor,
                                                        warm_up_epoch=opt.warm_up_epoch)
    #training_print = TrainPrint(steps_per_epoch, opt.epochs)
    print_lr = PrintLearningRate()

    return [tensorboard, learning_rate_scheduler, print_lr, checkpoint, early_stopping]
    


def train(opt):
    K.clear_session()
    video_input = Input(shape=(None, None, None, 3))
    model = nets.network[opt.network](video_input, num_classes=opt.num_classes)
    print("Create {} model with {} classes".format(opt.network, opt.num_classes))

    if opt.pretrained_weights is not None:
        model.load_weights(opt.pretrained_weights)
        print("Loading weights from {}".format(opt.pretrained_weights))

    optimizer = get_optimizer(opt)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])

    train_data_generator, train_num = ucf_dataset.get_ucf101(opt.video_path, opt.train_list, opt.name_path, 
                                                  'train', opt.batch_size, opt.num_classes, True, opt.short_side, 
                                                  opt.crop_size, opt.clip_len, opt.n_samples_for_each_video)
    val_data_generator, val_num = ucf_dataset.get_ucf101(opt.video_path, opt.val_list, opt.name_path, 'val', 
                                                opt.batch_size, opt.num_classes, False, opt.short_side, 
                                                opt.crop_size, opt.clip_len, opt.n_samples_for_each_video)
    
    callbacks = create_callbacks(opt, max(1, math.ceil(train_num/opt.batch_size)))

    model.fit_generator(train_data_generator, steps_per_epoch=max(1, math.ceil(train_num/opt.batch_size)),
                        epochs=opt.epochs, validation_data=val_data_generator, validation_steps=max(1, math.ceil(val_num/opt.batch_size)),
                        workers=opt.workers, callbacks=callbacks)
    model.save_weights(os.path.join(os.path.join(opt.root_path, opt.result_path), 'trained_weights_final.h5'))

    
if __name__=="__main__":
    opt = parse_opts()
    print(opt)
    train(opt)