import os
import json
from pathlib import Path 

import torch
import torch.nn as nn

from PerfPredRecV2.eval.model_architectures import load_model_def

# How many attributes each domain has
# Just to initiate class attributes
CLASS = {
    "cityscapes": 19,
    "kitti": 10,
}

class ModelWrapper(nn.Module):
    """
    A class to wrap a model with a preprocess function as well as a postprocess function.
    """
    def __init__(self, model, preprocessing=None, postprocessing=None) -> None:
        """
        :param model: The model to be wrapped
        :param preprocess: A callable preprocess function
        :param postprocess: A callable postprocess function
        """

        super().__init__()

        # Initialization
        self.model = model
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.device = next(model.parameters()).device

    def _remove_rec_output(self, x):
        assert isinstance(x, dict), "x is not a dictionary, which is expected"
        for key in x.keys():
            assert key in ['image_reconstruction', 'logits'], f"x has not the correct keys {x.keys()}"
        return x['logits']

    def _preprocess(self, x):
        """
        A function to perform preprocessing on the model input.

        :param x: Input image to the model.
        :return: Preprocessed image
        """
        if self.preprocessing is None:
            return x
        else:
            return self.preprocessing(x)

    def _postprocess(self, y):
        """
        A function to perform postprocessing on the model output.

        :param y: Output of the model.
        :return: Postprocessed output
        """
        # return self.remove_rec_output(y)
        if self.postprocessing is None:
            return y
        else:
            return self.postprocessing(y)

    def forward(self, x):
        x = self._preprocess(x)
        y = self.model(x)
        return self._postprocess(y)

    def remove_rec_output(self, x):
        assert isinstance(x, dict), "x is not a dictionary, which is expected"
        return x['logits']

# TODO: make an object of this
# so that it can become more flexible
def load_model(
    weights_path: str,
    load_pretrained=True,
    device=torch.device('cpu'),
    domain='cityscapes'
) -> nn.Module:
    assert os.path.exists(weights_path), f"Path does not exist."

    num_class = CLASS[domain]
    assert domain in CLASS, f"Specified domain `{domain}` is not supported."


    cfg_path = os.path.join(weights_path, "models", "opt.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = load_model_def(
        cfg["model_name"],
        num_class,
        backbone=cfg["encoder"],
        rec_decoder=cfg["rec_decoder"]
    )

    if load_pretrained:
        dict_path = [str(p) for p in Path(weights_path).rglob("*.pth")][0]
        model.load_state_dict(torch.load(dict_path, map_location=device))
        print("\nSegmentation weights are sucessfully loaded.")
    else:
        print("\nModel instantiated with random weights.")

    model = model.to(device)
    model.eval()
    return ModelWrapper(model, preprocessing=None)

