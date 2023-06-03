
from models import *
from train import *
from hddds import *

simple_net = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
simple_lstm = SimpleLSTM(in_shape=(3,24,24), out_shape=(10,))
my_ff = MyFastResA()

x = torch.rand(7,3,24,24)
xs = torch.rand(7,20,3,24,24)
xxs = torch.rand(7,3,256,256)

default_configs = TrainConfigs(num_epochs=3)

# train(my_ff, ["bottle", "pill"], default_configs)

# train on tabular data HAI, SWAT, WADI

def train(ds_name):
    data = get_dataset(ds_name)
    lstm = SimpleLSTM(in_shape=(data.num_features,), out_shape=(1,), return_mode="all")
    model = train_tabular(lstm, ds_name, data, default_configs)
    return model

if __name__== "__main__":
    hai =  "hai_sliding_100_train"
    swat = "swat_sliding_100_train"
    wadi = "wadi_sliding_100_train"
    m = train(hai)
    m1 = train(swat)
    m2 = train(wadi)
    breakpoint()
