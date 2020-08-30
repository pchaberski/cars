"""Architectures dictionary."""


from models.squeezenet import SqueezeNet_10, SqueezeNet_11
from models.squeezenext import (
    SqueezeNext23_10, SqueezeNext23_10_v5,
    SqueezeNext23_20, SqueezeNext23_20_v5)
from models.mobilenet import MobileNet_v2
from models.efficientnet import (
    EfficientNet_b0,
    EfficientNet_b1,
    EfficientNet_b2,
    EfficientNet_b3,
    EfficientNet_b4,
    EfficientNet_b5,
    EfficientNet_b6,
    EfficientNet_b7
)


def get_architectures_dictionary():
    arch_dict = {
        'sq_10': SqueezeNet_10,
        'sq_11': SqueezeNet_11,
        'sqnxt_1': SqueezeNext23_10,
        'sqnxt_1v5': SqueezeNext23_10_v5,
        'sqnxt_2': SqueezeNext23_20,
        'sqnxt_2v5': SqueezeNext23_20_v5,
        'mob_v2': MobileNet_v2,
        'eff_b0': EfficientNet_b0,
        'eff_b1': EfficientNet_b1,
        'eff_b2': EfficientNet_b2,
        'eff_b3': EfficientNet_b3,
        'eff_b4': EfficientNet_b4,
        'eff_b5': EfficientNet_b5,
        'eff_b6': EfficientNet_b6,
        'eff_b7': EfficientNet_b7
    }

    return arch_dict
