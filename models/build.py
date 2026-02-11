import torch

from models.oracle.modeling_oracle import ORACLEAD


def build_model(cfg):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "ORACLEAD": ORACLEAD,
    }

    if model_name in model_mapping:
        model = model_mapping[model_name](cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if torch.cuda.is_available():
        model = model.cuda()

    return model
