
from .dataset_utils import *
from .mvtec import *
from .squad import *
from .timeseries import *


def get_data_bundle(dataset_name, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == "mvtec":
        return get_mvtec_bundle(**kwargs)

    elif dataset_name == "squad":
        return get_squad_bundle(**kwargs)

    elif dataset_name == "hai":
      return get_timeseries_bundle("hai", **kwargs)

    elif dataset_name == "swat":
      return get_timeseries_bundle("swat", **kwargs)

    elif dataset_name == "wadi":
      return get_timeseries_bundle("wadi", **kwargs)

    else:
        raise NotImplementedError()


