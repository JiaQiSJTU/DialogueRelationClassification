#!/bin/bash
#

python -W ignore main.py  --stage test --model cnn_baseline \
    --num_class 4 \
    --learning_rate 1e-2 --batch_size 8 \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model cnn_baseline \
    --num_class 6 \
    --learning_rate 1e-2 --batch_size 8 \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model cnn_baseline \
    --num_class 13 \
    --learning_rate 1e-2 --batch_size 8 \
    --checkpoint checkpoint_path