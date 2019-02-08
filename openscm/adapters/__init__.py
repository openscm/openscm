from .hector import Hector
from .magicc6 import MAGICC6

def get_adapter(model):
    model_classes = {
        "MAGICC6": MAGICC6,
        # "MAGICC7": MAGICC7,
        "Hector": Hector,
        # "PH99": PH99,
    }

    return model_classes[model]
