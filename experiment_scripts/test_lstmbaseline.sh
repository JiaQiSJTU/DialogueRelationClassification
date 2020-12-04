#!/bin/bash
#

python -W ignore main.py  --stage test --model lstm_baseline \
    --num_class 4 \
    --learning_rate 3e-4 --batch_size 8 \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model lstm_baseline \
    --num_class 6 \
    --learning_rate 3e-4 --batch_size 8 \
    --checkpoint checkpoint_path

python -W ignore main.py  --stage test --model lstm_baseline \
    --num_class 13 \
    --learning_rate 3e-4 --batch_size 8 \
    --checkpoint checkpoint_path
