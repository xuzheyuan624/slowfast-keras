#!/bin/bash

python train.py --root_path=/home/xuzheyuan/PycharmProjects/slowfast-keras \
    --video_path=UCF-101_jpg \
    --name_path=ucfTrainTestlist/classInd.txt \
    --train_list=ucfTrainTestlist/trainlist01.txt \
    --val_list=ucfTrainTestlist/testlist01.txt \
    --workers=4 \
    --batch_size=4 \
    --crop_size=112 \
    --clip_len=32 \
    --short_side=[256,320]