#!/bin/bash
#
# fix token type id bugs

python -W ignore main.py --stage train --model cnn_baseline --random_seed 42 \
	--num_class 4 --default_root_dir trained_models/cnn_4_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 42 \
	--num_class 6 --default_root_dir trained_models/cnn_6_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 42 \
	--num_class 13 --default_root_dir trained_models/cnn_13_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 52 \
	--num_class 4 --default_root_dir trained_models/cnn_4_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 52 \
	--num_class 6 --default_root_dir trained_models/cnn_6_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 52 \
	--num_class 13 --default_root_dir trained_models/cnn_13_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 62 \
	--num_class 4 --default_root_dir trained_models/cnn_4_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 62 \
	--num_class 6 --default_root_dir trained_models/cnn_6_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
python -W ignore main.py --stage train --model cnn_baseline --random_seed 62 \
	--num_class 13 --default_root_dir trained_models/cnn_13_class \
	--learning_rate 1e-2 --batch_size 2 --max_epochs 32 --accumulate_grad_batches 16 \
	--device 2
