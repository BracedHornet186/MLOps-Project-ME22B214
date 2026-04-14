import torch
from models.ncmnet.config import get_config
from models.ncmnet.ncmnet import NCMNet


def load_model(weight_path: str) -> NCMNet:
    config, unparsed = get_config()
    print(f'Load NCMNet with the config: {config}')
    model = NCMNet(config)
    weights = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(weights['state_dict'])
    return model
