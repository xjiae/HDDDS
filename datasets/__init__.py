
from .dataset_utils import *
from .mvtec import *
from .squad import *
from .timeseries import *


def get_data_bundle(dataset_name, train_batch_size=4, test_batch_size=4, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == "mvtec":
        return get_mvtec_bundle(train_batch_size=train_batch_size, test_batch_size=test_batch_size, **kwargs)

    elif dataset_name == "squad":
        return get_squad_bundle(train_batch_size=train_batch_size, test_batch_size=test_batch_size, **kwargs)

    elif dataset_name == "hai":
      return get_timeseries_bundle("hai", train_batch_size=train_batch_size, test_batch_size=test_batch_size, **kwargs)

    elif dataset_name == "swat":
      return get_timeseries_bundle("swat", train_batch_size=train_batch_size, test_batch_size=test_batch_size, **kwargs)

    elif dataset_name == "wadi":
      return get_timeseries_bundle("wadi", train_batch_size=train_batch_size, test_batch_size=test_batch_size, **kwargs)

    else:
        raise NotImplementedError()


