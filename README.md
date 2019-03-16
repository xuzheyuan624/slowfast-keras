# slowfast-keras
A implementation of [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) with tf.keras
This code use half-period cosine schedule of learning rate decaying and warm-up strategy, but I don't kown if it's the same as the paper.

## Requirements
tensorflow >= 1.12</ br>
pillow>=5.1.0

## Train
### 1. Prepare the dataset
You can use UCF101 or other datasets, which should be orgnized as :

<!-- TOC -->autoauto- [slowfast-keras](#slowfast-keras)auto    - [Requirements](#requirements)auto    - [Train](#train)auto        - [1. Prepare the dataset](#1-prepare-the-dataset)auto        - [2. Change settings](#2-change-settings)autoauto<!-- /TOC -->

### 2. Change settings
You must change the root_path, video_path, name_path or others in train.sh for your own.

### 3. Train
Then you can train with:</ br>
'''
bash train.sh
'''