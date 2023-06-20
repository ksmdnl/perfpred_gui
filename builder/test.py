#!/usr/bin/env python

# FIXME: this is ugly
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import sys
import os
sys.path.append("../")
from PerfPredRecV2.models import wrapper

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

def inference(model):
    video_path = "videos/tbilisi.mov"
    cap = cv2.VideoCapture(video_path)

    preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (500, 150)) 
            # Preprocess the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame).unsqueeze(0)
            
            # Perform inference
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.squeeze().cpu().numpy()

            # Step 6: Visualize or Save the Results
            # Example: Display the segmentation mask on the frame
            mask = predicted.astype('uint8') * (255 // 19)  # Normalize mask values for visualization
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
            cv2.imshow("Semantic Segmentation", overlay)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # play()
    model_name = "SwiftNet"
    num_classes = 19
    backbone = "resnet18"
    model = wrapper.load_model_def(model_name, num_classes)
    weightspath = os.path.join("weights", model_name.lower(), backbone, 'model.pth')
    assert os.path.exists(weightspath), f"{weightspath} does not exists."
    model.load_state_dict(torch.load(weightspath, map_location=device))

    # model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    inference(model)
