#!/bin/bash

# List of dataset names
dataset_names=("wadi_all_sliding_100" "hai_sliding_100")



# List of explainers
explainers=("LIME" "SHAP")

# Nested for loops to iterate over all combinations
for ds_name in "${dataset_names[@]}"
    do
        for explainer in "${explainers[@]}"
        do
            echo "Running experiment for dataset: $ds_name, model: $model, explainer: $explainer"
            CUDA_VISIBLE_DEVICES=0,1,2 python run_model.py -ds "$ds_name" -exp "$explainer" -test_only 1 -batch_size_test 1
        done
    done
done

