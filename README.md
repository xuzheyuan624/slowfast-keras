# slowfast-keras
A implementation of [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) with tf.keras
This code use half-period cosine schedule of learning rate decaying and warm-up strategy, but I don't kown if it's the same as the paper.

## Requirements
tensorflow >= 1.12<br>
pillow>=5.1.0

## Train
### 1. Prepare the dataset
You can use UCF101 or other datasets, which should be orgnized as :

<!-- TOC -->
- UCF101
    - ApplyEyeMakeUp
    - ApplyLipstick
    - Archery
    - ......
- [ucfTrainTestlist](https://www.crcv.ucf.edu/wp-content/uploads/2019/03/UCF101TrainTestSplits-RecognitionTask.zip)
    - classInd.txt
    - trainlist01.txt
    - testlist01.txt
    - ......
<!-- /TOC -->

### 2. Change settings
You must change the root_path, video_path, name_path or others in train.sh for your own.

### 3. Train
Then you can train with:<br>
```
bash train.sh
```

## Contact
If you have questions or suggestions, you can cantact me at [xuzheyuan624@163.com](xuzheyuan624@163.com)
