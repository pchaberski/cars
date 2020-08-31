"""Architectures dictionary."""


from models.architectures.squeezenet import SqueezeNet_10, SqueezeNet_11
from models.architectures.squeezenext import (
    SqueezeNext23_10, SqueezeNext23_10_v5,
    SqueezeNext23_20, SqueezeNext23_20_v5)
from models.architectures.mobilenet import MobileNet_v2
from models.architectures.efficientnet import (
    EfficientNet_b0,
    EfficientNet_b1,
    EfficientNet_b2,
    EfficientNet_b3,
    EfficientNet_b4,
    EfficientNet_b5,
    EfficientNet_b6,
    EfficientNet_b7
)
from models.architectures.ghostnet import GhostNet
from models.architectures.hardnet import (
    HarDNet_39, HarDNet_39dw,
    HarDNet_68, HarDNet_68dw,
    HarDNet_85, HarDNet_85dw
)
from models.architectures.shufflenet import (
    ShuffleNet_v2_x05,
    ShuffleNet_v2_x10,
    ShuffleNet_v2_x15,
    ShuffleNet_v2_x20
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
        'eff_b7': EfficientNet_b7,
        'ghost': GhostNet,
        'hard_39': HarDNet_39,
        'hard_39dw': HarDNet_39dw,
        'hard_68': HarDNet_68,
        'hard_68dw': HarDNet_68dw,
        'hard_85': HarDNet_85,
        'hard_85dw': HarDNet_85dw,
        'shuff_v2_x05': ShuffleNet_v2_x05,
        'shuff_v2_x10': ShuffleNet_v2_x10,
        'shuff_v2_x15': ShuffleNet_v2_x15,
        'shuff_v2_x20': ShuffleNet_v2_x20
    }

    return arch_dict
