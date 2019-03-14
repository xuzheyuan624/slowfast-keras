import os
import random
import math
import copy
import numpy as np
from .spatial_transforms import RandomCrop, Scale, RandomHorizontalFlip, CenterCrop, Compose, Normalize, PreCenterCrop
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

def dataset(name, video_path, file_path, name_path, mode, batch_size, num_classes, shuffle, short_side=[256, 320], crop_size=224, clip_len=64, n_samples_for_each_video=1):
    if name == 'ucf101':
        video_files, label_files = get_ucf101(video_path, file_path, name_path, mode, num_classes)

    if mode == 'train':
        spatial_transforms = Compose([
            PreCenterCrop(),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            Normalize()
        ])
        temporal_transforms = TemporalRandomCrop(clip_len)
    elif mode == 'val':
        spatial_transforms = Compose([
            PreCenterCrop(),
            Scale(crop_size),
            Normalize()
        ])
        temporal_transforms = TemporalCenterCrop(clip_len)
    else:
        raise ValueError('mode must be train or val')

    dataset = makedataset(video_files, label_files, n_samples_for_each_video, clip_len)


    data_generator = get_dataset(dataset, batch_size, num_classes, shuffle, spatial_transforms=spatial_transforms, temporal_transforms=temporal_transforms)
    # elif mode == 'val':
    #     data_generator = get_val_dataset(dataset, batch_size, num_classes, spatial_transforms=spatial_transforms, temporal_transforms=temporal_transforms)
    # else:
    #     raise ValueError('mode must be train or val')
    
    return data_generator, len(dataset)

    
def makedataset(video_files, label_files, n_samples_for_each_video, clip_len):
    dataset = []
    for i, video_file in enumerate(video_files):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_files)))
        
        if not os.path.exists(video_file):
            print('{} is not exist'.format(video_file))
            continue
        
        n_frame_path = os.path.join(video_file, 'n_frames')
        n_frames = int(load_value_file(n_frame_path))

        if n_frames<=0:
            continue
        
        sample = {
            'video_path':video_file,
            'label':int(label_files[i])
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



def get_dataset(dataset, batch_size, num_classes, shuffle, spatial_transforms=None, temporal_transforms=None):
    i = 0
    n = len(dataset)
    while True:
        video_data = []
        label_data = []
        for b in range(batch_size):
            if i==0 and shuffle:
                random.shuffle(dataset)
            path = dataset[i]['video_path']
            frame_indices = dataset[i]['frame_indices']
            if temporal_transforms is not None:
                frame_indices = temporal_transforms(frame_indices)

            clip = load_clip_video(path, frame_indices)

            if spatial_transforms is not None:
                spatial_transforms.randomize_parameters()
                clip = [spatial_transforms(img) for img in clip]

            clip = np.stack(clip, 0)
            video_data.append(clip)
            label_data.append(dataset[i]['label'])
            i = (i + 1)%n
            if i == 0:
                break
        video_data = np.array(video_data)
        label_data = np.eye(num_classes)[label_data]
        yield video_data, label_data

def get_val_dataset(dataset, batch_size, num_classes, spatial_transforms=None, temporal_transforms=None):
    i = 0
    n = len(dataset)
    while i < n:
        video_data = []
        label_data = []
        for b in range(batch_size):
            path = dataset[i]['video_path']
            frame_indices = dataset[i]['frame_indices']
            if temporal_transforms is not None:
                frame_indices = temporal_transforms(frame_indices)

            clip = load_clip_video(path, frame_indices)

            if spatial_transforms is not None:
                spatial_transforms.randomize_parameters()
                clip = [spatial_transforms(img) for img in clip]

            clip = np.stack(clip, 0)
            video_data.append(clip)
            label_data.append(dataset[i]['label'])
            i = i + 1
            if i == n:
                break
        video_data = np.array(video_data)
        label_data = np.eye(num_classes)[label_data]
        yield video_data, label_data





if __name__=="__main__":
    import cv2
    data_generator = get_ucf101('../UCF-101_jpg', '../ucfTrainTestlist/trainlist01.txt', '../ucfTrainTestlist/classInd.txt', 'train', 4, 101)
    for video, label in data_generator:
        print(video.shape)
        print(label.shape)
        img = video[0, 0, :, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('test.jpg', img)
        raise Exception('')
