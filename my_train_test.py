
from models import *
from train import *
from hddds import *

simple_net = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
my_ff = MyFastResA()

x = torch.rand(7,3,24,24)
xs = torch.rand(7,20,3,24,24)
xxs = torch.rand(7,3,256,256)

hai_lstm = SimpleLSTM(in_shape=(86,), out_shape=(1,), return_mode="last")
swat_lstm = SimpleLSTM(in_shape=(51,), out_shape=(1,), return_mode="last")
wadi_lstm = SimpleLSTM(in_shape=(127,), out_shape=(1,), return_mode="last")

default_configs = TrainConfigs(num_epochs=3)
hai_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_HAI_LOADER_CONFIGS)
hai_sliding_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_HAI_SLIDING_LOADER_CONFIGS)

wadi_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_WADI_LOADER_CONFIGS)
wadi_sliding_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_WADI_SLIDING_LOADER_CONFIGS)

swat_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_SWAT_LOADER_CONFIGS)
swat_sliding_configs = TrainConfigs(num_epochs=3, loaders_kwargs = DEFAULT_SWAT_SLIDING_LOADER_CONFIGS)


# train(my_ff, ["bottle", "pill"], default_configs)

# train on tabular data HAI, SWAT, WADI

def xj_train(ds_name):
    data = get_dataset(ds_name)
    lstm = SimpleLSTM(in_shape=(data.num_features,), out_shape=(1,), return_mode="all")
    model = train_tabular(lstm, ds_name, data, default_configs)
    return model

if __name__== "__main__":
    pass
    '''
    hai =  "hai_sliding_100_train"
    swat = "swat_sliding_100_train"
    wadi = "wadi_sliding_100_train"
    m = xj_train(hai)
    m1 = xj_train(swat)
    m2 = xj_train(wadi)
    breakpoint()
    '''

