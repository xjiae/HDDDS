
from models import *
from train import *

simple_net = SimpleNet(in_shape=(3,24,24), out_shape=(10,))
simple_lstm = SimpleLSTM(in_shape=(3,24,24), out_shape=(10,))
my_ff = MyFastResA()

x = torch.rand(7,3,24,24)
xs = torch.rand(7,20,3,24,24)
xxs = torch.rand(7,3,256,256)

default_configs = TrainConfigs(num_epochs=3)

# train(my_ff, ["bottle", "pill"], default_configs)

