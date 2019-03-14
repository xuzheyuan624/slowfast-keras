
import random
import os
import cv2


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
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value
