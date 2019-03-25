# slowfast-keras
A implementation of [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) with tf.keras
This code use half-period cosine schedule of learning rate decaying and warm-up strategy, but I don't kown if it's the same as the paper.

## Requirements
tensorflow >= 1.12<br>
pillow>=5.1.0

##Get code
```
git clone https://github.com/xuzheyuan624/slowfast-keras.git
cd slowfast-keras
```

## Train
### 1. Download the dataset
You can use UCF101 or other datasets, which should be orgnized as :
<!-- TOC -->
- [UCF101](https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar)
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
### 2. Prepare dataset for training
convert video to jpgs with:
```
python utils/ucf_hmdb51_frames.py UCF101 UCF101_jpg
```
calculate the video's frames
```
python utils/ucf_hmdb51_frames.py UCF101_jpg
```

### 3. Change settings and Train
You must change the ```root_path, video_path, name_path``` or others in ```train.sh``` for your own. See details in ```opts.py```<br>
For example: 
```root_path```is ```path to slowfast-keras```<br>
```video_path```is```path to UCF101_jpg```<br>
```name_path```is ```path to classInd.txt```<br>
......<br>
Then you can train with:<br>
```
bash train.sh
```
## Code Reference
[1] [SlowFastNetworks](https://github.com/RI-CH/SlowFastNetworks)<br>
[2] [3D-ResNets-Pytorch](https://github.com/kenshohara/3D-ResNets-PyTorch)<br>
[3] [SGDR](https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452)
