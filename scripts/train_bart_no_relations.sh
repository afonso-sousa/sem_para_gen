#!/bin/bash

datasets_dir="processed-data"
output_dir="output"
model_name="facebook/bart-base"
lr="1e-4"
dataset="paws"
output_file=$model_name-$dataset-lr$lr-no-relations
file_suffix="with_graph"

python train.py \
    --model_name_or_path $model_name \
    --with_graph \
    --delimiters \
    --relations \
    --output_dir $output_dir/$output_file \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --dataset_name $datasets_dir/$dataset \
    --splits_suffix $file_suffix \
    --num_warmup_steps 100 \
    --learning_rate $lr \
    --num_train_epochs 4 \
    --max_eval_samples 10 \
    --evaluation_interval 2 \
    --max_source_length 256 \
    --max_target_length 64 \
