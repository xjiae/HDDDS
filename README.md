# Ground Truth eXplanation Dataset
1. Please run the `setup.sh` file to downlad the datasets.
2. Please run `conda env create --name hddds --file environment.yml` to install the necessary packages.
3. Please find the notebook `example.ipynb` for the sample access to our dataset and potential use of dataloaders.
4. Please find the code for experiments in `hddds.py` (which is going to be slow). The detailed code is consist of three parts, and separated in three files:
-  `experiments/train.py`: training the predictive models;
-  `experiments/get_explanations.py`: generating the explanations using different feature atrribution methods;
-  `experiments/summary.py`: evaluating the performance of the explanations against ground truth.

PS: If you ever wonder why there are so many 'd's in `hddds`, it is pinyin for `Hao Da De DataSet` (very large dataset) :)
