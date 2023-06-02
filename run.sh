#!/bin/bash

# List of dataset names
dataset_names=("swat_all_sliding_100" "wadi_all_sliding_100")



# List of explainers
explainers=("IG")

# Nested for loops to iterate over all combinations
for ds_name in "${dataset_names[@]}"
do
    for model in "${models[@]}"
    do
        for explainer in "${explainers[@]}"
        do
            echo "Running experiment for dataset: $ds_name, model: $model, explainer: $explainer"
            python run_model.py -ds "$ds_name" -exp "$explainer"
        done
    done
done

