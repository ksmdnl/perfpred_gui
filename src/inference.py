#!/usr/bin/env python
import os
import cv2
import warnings
import importlib
import random

import numpy as np

import torch
import torchvision.transforms as transforms

from src.monitor import PerceptionMonitor

from robustness.attacks import ATTACKS
from dataloader.definitions.labels_file import dataset_labels

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MAX_MIOU = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
mean = torch.tensor(MEAN).reshape(1, -1, 1, 1)
std = torch.tensor(STD).reshape(1, -1, 1, 1)

# do_resize = os.getenv("RESIZE", True)
do_resize = False
factor = 1
HEIGHT, WIDTH = int(180 * factor), int(390 * factor)
# HEIGHT, WIDTH = 300, 1000

preprocess = []
preprocess.append(transforms.ToPILImage())
if do_resize:
    preprocess.append(transforms.Resize((HEIGHT, WIDTH)))
preprocess.append(transforms.ToTensor())
preprocess.append(
    transforms.Normalize(mean=MEAN, std=STD)
)
preprocess = transforms.Compose(preprocess)

COEFF = np.array([0.09334191 -0.6773528  -9.46617536])
COEFF = COEFF[-1:]

monitor = PerceptionMonitor()

def get_miou_estimate(psnr):
    # miou = 0
    # for i, coeff in enumerate(COEFF):
    #     miou += coeff * (psnr ** i)
    # return miou

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
import matplotlib.pyplot as plt

psnr_history = []
miou_history = []
frame_history = []

plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))

psnr_line, = ax.plot([], [], label="PSNR (dB)", color="blue")
miou_line, = ax.plot([], [], label="Estimated mIoU (%)", color="orange")

ax.set_xlabel("Frame")
ax.set_ylabel("Value")
ax.set_title("Reconstruction Quality")
ax.legend()
ax.grid(True)

def run_inference(model, path="builder/videos/tbilisi.mov"):
    cap = cv2.VideoCapture(path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        input_tensor = preprocess(frame).unsqueeze(0)

        apply_attack = False
        # apply_attack = bool(random.getrandbits(1))
        # if apply_attack:
        #     input_tensor, _ = attack(input_tensor, model.model)

        with torch.no_grad():
            output = model(input_tensor)

        output = postprocess(output)

        mask = output[0]

        if output[1] is not None:
            reconstructed = output[1]
            input_tensor_denorm = denormalize(input_tensor)

            psnr = get_psnr(input_tensor_denorm, reconstructed)
            miou = get_miou_estimate(psnr) / MAX_MIOU * 100
            status = monitor.update(psnr=psnr, miou=miou)
            print(status)

            # print(f"Apply attack: {apply_attack} PSNR: {psnr:.2f} dB, mIoU estimate: {miou:.2f} %")

            # Store history
            frame_history.append(frame_idx)
            psnr_history.append(psnr)
            miou_history.append(miou)

            # Update plot
            psnr_line.set_data(frame_history, psnr_history)
            miou_line.set_data(frame_history, miou_history)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            cv2.imshow("Reconstruction", reconstructed)

        cv2.imshow("Ground truth", frame)
        cv2.imshow("Segmentation", mask)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    plt.close(fig)