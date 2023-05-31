#!/bin/bash

# List of dataset names
dataset_names=("swat_all" "wadi_all")

# List of models
models=("LR" "SVM")

# List of explainers
explainers=("SHAP" "LIME")

# Nested for loops to iterate over all combinations
for ds_name in "${dataset_names[@]}"
do
    for model in "${models[@]}"
    do
        for explainer in "${explainers[@]}"
        do
            echo "Running experiment for dataset: $ds_name, model: $model, explainer: $explainer"
            python experiment.py -ds_name "$ds_name" -model "$model" -explainer "$explainer"
        done
    done
done
