import os
import random
import math
import copy
import numpy as np
from tensorflow.keras.utils import Sequence
from .spatial_transforms import RandomCrop, Scale, RandomScale, RandomHorizontalFlip, CenterCrop, Compose, Normalize
from .tempora_transforms import TemporalRandomCrop, TemporalCenterCrop
from .utils import load_value_file, load_clip_video


def get_ucf101(video_path, file_path, name_path, mode, num_classes):
    name2index = {}

    lines = open(name_path, 'r').readlines()
    for i, class_name in enumerate(lines):
        class_name = class_name.split()[1]
        name2index[str(class_name)]=i
    
    assert num_classes == len(name2index)

    video_files = []
    label_files = []
    for path_label in open(file_path, 'r'):
        if mode == 'train':
            path, _ = path_label.split()
        elif mode == 'val':
            path = path_label
        else:
            raise ValueError('mode must be train or val')
        pathname, _ = os.path.splitext(path)
        video_files.append(os.path.join(video_path, pathname))
        label = pathname.split('/')[0]
        label_files.append(name2index[label])
    return video_files, label_files


class DataGenerator(Sequence):
    def __init__(self, data_name, video_path, file_path, 
                 name_path, mode, batch_size, num_classes, 
                 shuffle, short_side=[256, 320], crop_size=224, 
                 clip_len=64, n_samples_for_each_video=1):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        if data_name == 'ucf101':
            self.video_files, self.label_files = get_ucf101(video_path, file_path, name_path, mode, num_classes)

        if mode == 'train':
            self.spatial_transforms = Compose([
                RandomScale(short_side),
                RandomCrop(crop_size),
                RandomHorizontalFlip(),
                Normalize()
            ])
            self.temporal_transforms = TemporalRandomCrop(clip_len)
        elif mode == 'val':
            self.spatial_transforms = Compose([
                Scale(crop_size),
                CenterCrop(crop_size),
                Normalize()
            ])
            self.temporal_transforms = TemporalCenterCrop(clip_len)
        else:
            raise ValueError('mode must be train or val')
        
        self.dataset = self.makedataset(n_samples_for_each_video, clip_len)
        if self.shuffle:
            random.shuffle(self.dataset)

    def __len__(self):
        return math.ceil(len(self.video_files)/self.batch_size)

    def __getitem__(self, index):
        batch_dataset = self.dataset[index*self.batch_size:(index+1)*self.batch_size]
        video_data, label_data = self.data_generator(batch_dataset)
        return video_data, label_data

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.dataset)

    def makedataset(self, n_samples_for_each_video, clip_len):
        dataset = []
        for i, video_file in enumerate(self.video_files):
            if i % 1000 == 0:
                print('dataset loading [{}/{}]'.format(i, len(self.video_files)))
            
            if not os.path.exists(video_file):
                print('{} is not exist'.format(video_file))
                continue
            
            n_frame_path = os.path.join(video_file, 'n_frames')
            n_frames = int(load_value_file(n_frame_path))

            if n_frames<=0:
                continue
            
            sample = {
                'video_path':video_file,
                'label':int(self.label_files[i])
            }
            if n_samples_for_each_video == 1:
                sample['frame_indices'] = list(range(1, n_frames+1))
                dataset.append(sample)

            else:
                if n_samples_for_each_video > 1:
                    step = max(1, math.ceil((n_frames - 1 - clip_len) / (n_samples_for_each_video - 1)))
                else:
                    step = clip_len
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(j, min(n_frames + 1, j + clip_len)))
                    dataset.append(sample_j)

        return dataset

    def data_generator(self, batch_dataset):
        video_data = []
        label_data = []
        for data in batch_dataset:
            path = data['video_path']
            frame_indices = data['frame_indices']
            if self.temporal_transforms is not None:
                frame_indices = self.temporal_transforms(frame_indices)

            clip = load_clip_video(path, frame_indices)

            if self.spatial_transforms is not None:
                self.spatial_transforms.randomize_parameters()
                clip = [self.spatial_transforms(img) for img in clip]

            clip = np.stack(clip, 0)
            video_data.append(clip)
            label_data.append(data['label'])
        video_data = np.array(video_data)
        label_data = np.eye(self.num_classes)[label_data]
        return video_data, label_data
