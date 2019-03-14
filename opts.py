import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default=None,
        type=str,
        help='Root directory path of data'
    )
    parser.add_argument(
        '--video_path',
        default='UCF-101_jpg',
        type=str,
        help='Directory of videos'
    )
    parser.add_argument(
        '--name_path',
        default=None,
        type=str,
        help='Directory of classes name'
    )
    parser.add_argument(
        '--train_list',
        default=None,
        type=str,
        help='Path to training list'
    )
    parser.add_argument(
        '--val_list',
        default=None,
        type=str,
        help='Path to validation list'
    )
    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Directory of results'
    )
    parser.add_argument(
        '--data_name',
        default='ntu',
        type=str,
        help='Name of dataset'
    )
    parser.add_argument(
        '--gpus',
        default=[0],
        type=int,
        nargs='+',
        help='GPUs for running'
    )
    parser.add_argument(
        '--log_dir',
        default='log',
        type=str,
        help='Path to save log'
    )
    parser.add_argument(
        '--num_classes',
        default=101,
        type=int,
        help='Number of classes'
    )
    parser.add_argument(
        '--crop_size',
        default=224,
        type=int,
        help='Size of crop image input'
    )
    parser.add_argument(
        '--clip_len',
        default=64,
        type=int,
        help='Length of videos'
    )
    parser.add_argument(
        '--short_side',
        default=[256, 320],
        type=int,
        nargs='+',
        help='Short side of the image'
    )
    parser.add_argument(
        '--n_samples_for_each_video',
        default=1,
        type=int,
        help='Number of samples of each video'
    )
    parser.add_argument(
        '--lr',
        default=1.6,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum'
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-4,
        type=float,
        help='Weight decay'
    )
    parser.add_argument(
        '--lr_decay',
        default=0.8,
        type=float,
        help='Decay rate of learning rate'
    )
    parser.add_argument(
        '--cycle_length',
        default=10,
        type=int,
        help='Epochs to restart cycle when using SGDR'
    )
    parser.add_argument(
        '--multi_factor',
        default=1.5,
        type=float,
        help='Increasing rate of cycle length'
    )
    parser.add_argument(
        '--warm_up_epoch',
        default=5,
        type=int,
        help='Using warmup at first several epochs'
    )
    parser.add_argument(
        '--optimizer',
        default='SGD',
        type=str,
        help='Optimizer'
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        default=100,
        type=str,
        help='Epochs fot training'
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='Worker for loading data'
    )
    parser.add_argument(
        '--network',
        default='resnet50',
        type=str,
        help='Network'
    )
    parser.add_argument(
        '--pretrained_weights',
        default=None,
        type=str,
        help='Path to pre-trained model'
    )

    args = parser.parse_args()

    return args
