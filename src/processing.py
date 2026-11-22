import numpy as np

import torch
import torchvision.transforms as transforms

from dataloader.definitions.labels_file import dataset_labels
trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
mean = torch.tensor(MEAN).reshape(1, -1, 1, 1)
std = torch.tensor(STD).reshape(1, -1, 1, 1)

do_resize = False
preprocess = []
preprocess.append(transforms.ToPILImage())
factor = 1
# HEIGHT, WIDTH = int(360 * factor), int(750 * factor)
# if do_resize:
#     preprocess.append(transforms.Resize((HEIGHT, WIDTH)))
preprocess.append(transforms.ToTensor())
preprocess.append(
    transforms.Normalize(mean=MEAN, std=STD)
)
preprocess = transforms.Compose(preprocess)

COEFF = np.array([0.09334191 -0.6773528  -9.46617536])
COEFF = COEFF[-1:]

def colorize(tensor, num_classes=19):
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(num_classes):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor

def denormalize(tensor):
    img = ((tensor * std) + mean)
    img = img.squeeze().permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img

def postprocess(output, assign_colour=True):

    def get_mask(logits, assign_colour=True):
        _, mask = torch.max(logits, 1)
        mask = mask.squeeze().cpu().numpy()
        if assign_colour:
            # mask = colorize(mask).numpy() / 255.
            mask = colorize(mask).numpy().astype(np.uint8)
        return mask

    # output of single head model
    if type(output) is not dict: return (get_mask(output, assign_colour=assign_colour), None)

    mask = get_mask(output["logits"], assign_colour=assign_colour)
    rec_image = output["image_reconstruction"]
    rec_image = denormalize(rec_image)
    return (mask, rec_image)

def colorize(tensor, num_classes=19):
    new_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=torch.uint8)
    for trainid in range(num_classes):
        new_tensor[tensor == trainid] = torch.tensor(trainid2label[trainid].color, dtype=torch.uint8)
    new_tensor[tensor == -100] = torch.tensor(trainid2label[255].color, dtype=torch.uint8)
    return new_tensor