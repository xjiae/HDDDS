

import warnings
warnings.filterwarnings("ignore")
from train import *
# from explainers.shap import SHAP
from explainers.lime import LIME
# import shap
# from openxai import Explainer
def main():

    with open("models/LSTM/StackedLSTM.pt", "rb") as f:
            SAVED_MODEL = torch.load(f)


    model = StackedLSTM()
    new = {k.replace('module.',''):v for k,v in SAVED_MODEL["state"].items()}
    model.load_state_dict(new)
    test_dataset = get_dataset('hai_test')
    model.eval().cuda()
    
    
    # shap = SHAP(model)
    lime = LIME(model)
    
    dataloader = DataLoader(test_dataset, batch_size=2024)
    ts, dist, att = [], [], []
    # with torch.no_grad():
    #     for batch in tqdm(dataloader):
    #         x = batch["x"].cuda()
    #         y = batch["y"].cuda()

    #         y_shap = shap.fit(x)
    #         breakpoint()
    batch  = next(iter(dataloader))
    x = batch['x'].cuda()
    background = x[:100]
    test_images = x[100:103]
    
    breakpoint()    
    lime.fit(x)
    # e = shap.DeepExplainer(model, background)
    # shap_values = e.shap_values(test_images)
    breakpoint()
        

if __name__ == "__main__":
    main()