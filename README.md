# Ground Truth eXplanation Dataset
1. Please run the `setup.sh` file to downlad the datasets.
2. Please run `pip install -r requirements.txt` to install the necessary packages.
3. Please find the notebook `example.ipynb` for the sample access to our dataset and potential use of dataloaders.
4. Please find the code for experiments in `hddds.py` (which is going to be slow). The detailed code is consist of three parts, and separated in three files:
-  `train.py`: training the predictive models;
-  `get_explanations.py`: generating the explanations using different feature atrribution methods;
-  `summary.py`: evaluating the performance of the explanations against ground truth.
