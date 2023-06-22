#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import sys
import os
from dataloader.definitions.labels_file import *
# FIXME: this is ugly
sys.path.append("../")
from PerfPredRecV2.models.wrapper import load_model_def

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
mean = torch.tensor(MEAN).reshape(1, -1, 1, 1)
std = torch.tensor(STD).reshape(1, -1, 1, 1)

HEIGHT, WIDTH = 150, 500
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame).unsqueeze(0)
            
            # Perform inference
            output = model(input_tensor)
            output = postprocess(output)
            mask = output[0]
            reconstructed = output[1]

            input_tensor = denormalize(input_tensor)
            print(f"PSNR: {get_psnr(input_tensor, reconstructed)} dB")
            cv2.imshow("Reconstruction", reconstructed)
            cv2.moveWindow("Reconstruction", 0, 200)
            cv2.imshow("Ground truth", frame)
            cv2.moveWindow("Segmentation", 0, 400)
            cv2.imshow("Segmentation", mask)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

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
    inference(model)
    # sample = cv2.imread('samples/munich_000068_000019_leftImg8bit.png')
    # sample = cv2.resize(sample, (WIDTH, HEIGHT))
    # tensor = preprocess(sample).unsqueeze(0)
    # with torch.no_grad():
    #     out = model(tensor)["image_reconstruction"]
    #     out = (out * std) + mean
    #     # tensor = (tensor * std) + mean
    #     print(get_psnr(tensor, out))
    #     out = out.squeeze(0).permute(1,2,0).cpu().numpy()
    # plt.imshow(tensor.squeeze(0).permute(1,2,0).cpu().numpy())
    # plt.show()