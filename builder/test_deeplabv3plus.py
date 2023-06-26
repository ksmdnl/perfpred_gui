#!/usr/bin/env python
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import sys
import os
from dataloader.definitions.labels_file import *
from PerfPredRecV2.models.wrapper import load_model_def

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

HEIGHT, WIDTH = 150, 500

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

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
    output = model(sample_tensor)["out"]
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
            # Preprocess the frame
            frame = cv2.resize(frame, (WIDTH, HEIGHT)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame).unsqueeze(0)
            print(input_tensor.shape)
            
            # Perform inference
            output = model(input_tensor)["out"]
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Step 6: Visualize or Save the Results
            # Example: Display the segmentation mask on the frame
            mask = predicted.astype('uint8') * (255 // 19)  # Normalize mask values for visualization
            # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            # mask = colorize(predicted).numpy() / 255.
            overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            print(mask.shape, predicted.shape)
            cv2.imshow("Semantic Segmentation", overlay)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    # model.classifier[-1] = torch.nn.Conv2d(256, 19, kernel_size=(1, 1))
    model = model.to(device)
    model.eval()
    inference(model)
    # single_frame(model)
