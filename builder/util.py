#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import cv2
import sys
import os
from dataloader.definitions.labels_file import *
from .attack import Attack

from PerfPredRecV2.models.wrapper import load_model_def

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
mean = torch.tensor(MEAN).reshape(1, -1, 1, 1)
std = torch.tensor(STD).reshape(1, -1, 1, 1)

HEIGHT, WIDTH = 1024, 2048
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

reg_coeff = np.array([0.09334191 -0.6773528  -9.46617536])

def get_miou_estimate(psnr):
    return 0.09334191 * (psnr**2) - 0.6773528 * psnr - 9.46617536

def colorize(tensor, num_classes=18):
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(num_classes):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor

def postprocess(output):
    def get_mask(logits):
        _, mask = torch.max(logits, 1)
        mask = mask.squeeze().cpu().numpy()
        return mask
    if type(output) is not dict: return get_mask(output)
    mask = get_mask(output["logits"])
    mask = colorize(mask).numpy() / 255.
    rec_image = output["image_reconstruction"]
    rec_image = denormalize(rec_image)
    return (mask, rec_image)

def generate_attack(epsilon, model, attack_type="fgsm", iterations=40):
    return Attack(epsilon, model, type=attack_type, iterations=iterations)

def apply(var, obj):
    return obj(var)

def get_psnr(img, gen_img):
    # Compute MSE first
    mse = ((img - gen_img) ** 2).mean()

    # Compute PSNR with MSE
    if isinstance(mse, torch.Tensor):
        psnr = 10 * torch.log10(1 / mse)
    elif isinstance(mse, np.float32):
        psnr = 10 * np.log10(1 / mse)
    return psnr

def denormalize(tensor):
    return ((tensor * std) + mean).squeeze().permute(1,2,0).cpu().numpy()

def display():
    pass

def inference(model):
    video_path = "videos/tbilisi.mov"
    cap = cv2.VideoCapture(video_path)
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (WIDTH, HEIGHT)) 
            # Preprocess the frame
            input_tensor = preprocess(frame).unsqueeze(0)
            
            # Perform inference
            output = model(input_tensor)
            output = postprocess(output)
            mask = output[0]
            reconstructed = output[1]

            input_tensor = denormalize(input_tensor)
            psnr = get_psnr(input_tensor, reconstructed)
            print(f"PSNR: {psnr} dB")
            print(f"mIoU estimate: {get_miou_estimate(psnr)} %")
            cv2.imshow("Reconstruction", reconstructed)
            cv2.moveWindow("Reconstruction", 0, 200)
            cv2.imshow("Ground truth", frame)
            cv2.moveWindow("Segmentation", 0, 400)
            cv2.imshow("Segmentation", mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def single_frame(model=None, sample="samples/munich_000068_000019_leftImg8bit.png", inference=False):
    sample = cv2.imread(sample)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample_tensor = preprocess(sample).unsqueeze(0)
    if inference:
        output = model(sample_tensor)["out"]
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()
        mask = colorize(predicted).numpy() / 255.
        cv2.imshow("Semantic Segmentation", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return sample_tensor

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
        if self.postprocessing is None:
            return y
        else:
            return self.postprocessing(y)

    def forward(self, x):
        x = self._preprocess(x)
        y = self.model(x)
        return self._postprocess(y)


def remove_rec_output(x):
    assert isinstance(x, dict), "x is not a dictionary, which is expected"
    for key in x.keys():
        assert key in ['image_reconstruction', 'logits'], f"x has not the correct keys {x.keys()}"
    return x['logits']

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "SwiftNetRec"
    num_classes = 19
    backbone = "resnet18"
    rec_decoder = "swiftnet"
    model = load_model_def(model_name, num_classes, rec_decoder=rec_decoder)
    weightspath = os.path.join("weights", model_name.lower(), backbone, 'model.pth')
    assert os.path.exists(weightspath), f"{weightspath} does not exists."
    model.load_state_dict(torch.load(weightspath, map_location=device))

    model = model.to(device)
    model.eval()
    # inference(model)
    model = ModelWrapper(model, postprocessing=remove_rec_output)
    attack = generate_attack(10 / 255, model, attack_type="metzen", iterations=1)
    sample = single_frame(inference=False)
    sample_og = denormalize(sample)
    print(sample_og)
    # cv2.imshow("Attacked image", sample_og)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    out_attack, _ = attack.generate(sample)
    print(out_attack)
    # out_attack = denormalize(out_attack)
    # cv2.imshow("Attacked image", out_attack)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()