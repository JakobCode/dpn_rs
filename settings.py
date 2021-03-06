from enum import Enum, auto

ucm_root_path = "./datasets/ucm/Images"
aid_root_path = "./datasets/aid"

class Dataset(Enum):
    AID = auto(),
    UCM = auto(),
    LCZ42 = auto(),
    LCZ42_val = auto(),
    LCZ42_regions = auto()


class Models(Enum):
    ResNet50 = auto()
    ResNet101 = auto()
    ResNet152 = auto()
    VGG16 = auto()
    VGG19 = auto()
    LCZ_BaseLine = auto()


class Approaches(Enum):
    dpn_rs = auto()
    prior_kl_forward = auto()
    prior_kl_reverse = auto()
    dpn_plus = auto()
    enn_cross_entropy = auto()


class ProblemType(Enum):
    class_split_1 = auto()
    class_split_2 = auto()
    class_split_3 = auto()                            