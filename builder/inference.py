#!/usr/bin/env python
import cv2
import warnings
import importlib

import numpy as np

import torch
import torchvision.transforms as transforms

from robustness.attacks import ATTACKS
from dataloader.definitions.labels_file import dataset_labels

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
mean = torch.tensor(MEAN).reshape(1, -1, 1, 1)
std = torch.tensor(STD).reshape(1, -1, 1, 1)

factor = 1
HEIGHT, WIDTH = int(180 * factor), int(390 * factor)
# HEIGHT, WIDTH = 300, 1000
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

reg_coeff = np.array([0.09334191 -0.6773528  -9.46617536])


def get_miou_estimate(psnr):
    return 0.09334191 * (psnr**2) - 0.6773528 * psnr + 9.46617536

def colorize(tensor, num_classes=19):
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(num_classes):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor

def postprocess(output, assign_colour=True):

    def get_mask(logits, assign_colour=True):
        _, mask = torch.max(logits, 1)
        mask = mask.squeeze().cpu().numpy()
        if assign_colour:
            mask = colorize(mask).numpy() / 255.
        return mask

    # output of single head model
    if type(output) is not dict: return (get_mask(output, assign_colour=assign_colour), None)

    mask = get_mask(output["logits"], assign_colour=assign_colour)
    rec_image = output["image_reconstruction"]
    rec_image = denormalize(rec_image)
    return (mask, rec_image)

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

attack_name = "metzen"
attack = ATTACKS
module = importlib.import_module(ATTACKS[attack_name].classpath)
attack = getattr(module, ATTACKS[attack_name].classname)
attack = attack(
    epsilon=10 / 255,
    alpha=1 / 255,
    iterations=20,
    target_type='dynamic',
    height=HEIGHT,
    width=WIDTH
)

def run_inference(model, path="builder/videos/tbilisi.mov"):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT)) 
        input_tensor = preprocess(frame).unsqueeze(0)
        # input_tensor, _ = attack(input_tensor, model.model)
            
        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        output = postprocess(output)

        mask = output[0]
        if output[1] is not None:
            reconstructed = output[1]
            input_tensor = denormalize(input_tensor)
            psnr = get_psnr(input_tensor, reconstructed)
            print(f"PSNR: {psnr} dB")
            print(f"mIoU estimate: {get_miou_estimate(psnr)} %")
            cv2.imshow("Reconstruction", reconstructed)
            cv2.moveWindow("Reconstruction", 200, 200)

        cv2.imshow("Ground truth", frame)
        cv2.moveWindow("Segmentation", 400, 200)
        cv2.imshow("Segmentation", mask)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()