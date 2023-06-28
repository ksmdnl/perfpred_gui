#!/usr/bin/env python

import numpy as np
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

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((768, 768)),
    # transforms.Resize((1024, 2048)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
HEIGHT, WIDTH = 1024, 2048

def play():
    file = "videos/tbilisi.mov"
    cap = cv2.VideoCapture(file)
    while(cap.isOpened()):
        result, frame = cap.read()
        if result:
            frame = cv2.resize(frame, (900, 300))
            cv2.imshow('VIDEO', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def colorize(tensor, num_classes=18):
    # tensor = tensor
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(num_classes):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor

def single_frame(model):
    sample = "samples/munich_000068_000019_leftImg8bit.png"
    sample = cv2.imread(sample)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    sample_tensor = preprocess(sample).unsqueeze(0)
    output = model(sample_tensor)
    _, predicted = torch.max(output, 1)
    predicted = predicted.squeeze().cpu().numpy()
    mask = colorize(predicted).numpy() / 255.
    print(mask.shape)
    cv2.imshow("Semantic Segmentation", mask)
    cv2.waitKey(0) # wait for ay key to exit window
    cv2.destroyAllWindows() # close all windows

def inference(model):
    video_path = "videos/tbilisi.mov"
    cap = cv2.VideoCapture(video_path)

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (HEIGHT, WIDTH)) 
            # Preprocess the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame).unsqueeze(0)
            
            # Perform inference
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Step 6: Visualize or Save the Results
            # Example: Display the segmentation mask on the frame
            mask = colorize(predicted).numpy() / 255.
            # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Semantic Segmentation", mask)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()