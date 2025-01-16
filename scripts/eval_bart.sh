#!/bin/bash

model_name="facebook/bart-base"
data_dir="processed-data"
dataset="paranmt-small" # paranmt-small, qqppos, paws
lr="1e-4"
checkpoint=$model_name-$dataset-lr$lr-standard
model_dir=output/${checkpoint}
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
splits_suffix="with_graph"

python evaluation.py \
    --model_name_or_path $model_dir \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file