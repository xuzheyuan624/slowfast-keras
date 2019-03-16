
import random
import os
from PIL import Image


def load_clip_video(video_path, frame_indices):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(load_image(image_path))
        else:
            return video
    return video


def load_image(image_path):
    with open(image_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value
