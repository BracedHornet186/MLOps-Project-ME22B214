from models.openglue.models.features.hardnet import GFTTAffNetHardNet
from models.openglue.models.features.opencv import methods as OPENCV_METHODS
from models.openglue.models.features.sift import SIFT
#from models.openglue.models.features.superpoint import methods as SUPERPOINT_METHODS
from models.openglue.models.features.iterative_features_extractor import IterativeLocalFeature
#from models.openglue.models.features.aliked import ALIKEDFeature
from functools import partial

methods = {
    'SIFT': SIFT,
    'GFTTAffNetHardNet': GFTTAffNetHardNet,
    #'ALIKED': ALIKEDFeature,
    #'MTLDesc': MTLDescFeature,
    #'PoSFeat': PoSFeatFeature,
    #**SUPERPOINT_METHODS,
    **OPENCV_METHODS
}


def get_feature_extractor(model_name):
    """
    Create method form configuration
    """
    if model_name not in methods:
        raise NameError('{} module was not found among local descriptors. Please choose one of the following '
                        'methods: {}'.format(model_name, ', '.join(methods.keys())))

    return methods[model_name]


__all__ = ['get_feature_extractor']
