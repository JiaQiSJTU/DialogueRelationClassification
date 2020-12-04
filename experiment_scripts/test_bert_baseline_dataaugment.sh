#!/bin/bash
#

python -W ignore main.py  --stage test --model bert_baseline \
    --num_class 4 \
    --learning_rate 1e-6 --batch_size 8 --data_augmentation True \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model bert_baseline \
    --num_class 6 \
    --learning_rate 1e-6 --batch_size 8 --data_augmentation True \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model bert_baseline \
    --num_class 13 \
    --learning_rate 1e-6 --batch_size 8 --data_augmentation True \
    --checkpoint checkpoint_path