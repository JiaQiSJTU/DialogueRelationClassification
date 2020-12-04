#!/bin/bash
#
# fix token type id bugs

python -W ignore main.py --stage train --model bert_baseline --random_seed 42\
    --num_class 4 --default_root_dir trained_models/bert_4_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 42\
    --num_class 6 --default_root_dir trained_models/bert_6_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 42\
    --num_class 13 --default_root_dir trained_models/bert_13_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 52\
    --num_class 4 --default_root_dir trained_models/bert_4_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 52\
    --num_class 6 --default_root_dir trained_models/bert_6_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 52\
    --num_class 13 --default_root_dir trained_models/bert_13_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 62\
    --num_class 4 --default_root_dir trained_models/bert_4_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 62\
    --num_class 6 --default_root_dir trained_models/bert_6_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
python -W ignore main.py --stage train --model bert_baseline --random_seed 62\
    --num_class 13 --default_root_dir trained_models/bert_13_class \
    --learning_rate 1e-6 --batch_size 1 \
    --max_epochs 32 --accumulate_grad_batches 32 \
    --device 0
