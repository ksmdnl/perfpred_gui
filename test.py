import dataloader.file_io.get_path as gp

#!/usr/bin/env python
import os
import sys
import torch
from builder.util import load_model, ModelWrapper, inference

from dataloader.pt_data_loader.specialdatasets import StandardDataset
import dataloader.pt_data_loader.mytransforms as mytransforms
from dataloader.definitions.labels_file import labels_cityscape_seg
from torch.utils.data import DataLoader

# How many attributes each domain has
# Just to initiate class attributes
CLASS = {
    "cityscapes": 19,
    "kitti": 10,
}

DEVICE = os.getenv("DEVICE", "cpu")


def iterate(dataset, model, metric, keys_to_load=['color', 'segmentation_trainid']):
    with torch.no_grad():
        for (_, inputs) in enumerate(dataset):
            im = inputs[("color_aug", 0, 0)]
            targets = inputs[(keys_to_load[-1], 0, 0)][:, 0, :, :].long()
            output = model(im)
            break
        seg_map = torch.argmax(output["logits"], dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        metric.update(targets, seg_map)
        miou = metric.get_scores()['meaniou']

if __name__ == "__main__":
    path = "./samples/cs_sec_101.mp4"
    domain = "cityscapes"
    # <SEGMENTATION_ARCH>_<BACKBONE_ARCH>
    model_name = "swiftnetrec_rn50"
    weights_path = os.path.join("weights", model_name)
    seg_arch, backbone_arc = model_name.split("_")
    num_class = CLASS[domain]

    assert os.path.exists(weights_path), f"Path does not exist."
    assert domain in CLASS, f"Specified domain `{domain}` is not supported."

    model = load_model(
        num_class=num_class,
        weights_path=weights_path,
        device=DEVICE,
    )

    model = ModelWrapper(model, postprocessing=None)

    inference(model, path=path)
    sys.exit(0)

    # labels = labels_cityscape_seg.getlabels()  # original labels used by Cityscapes
    # keys_to_load = ['color', 'segmentation']
    # subset = 'lindau'
    # # dataset = 'cityscapes_sequence'
    # dataset = 'cityscapes'

    # val_data_transforms = [mytransforms.CreateScaledImage()]
    # val_data_transforms.append(mytransforms.CreateColoraug())  # Adjusts keys so that NormalizeZeroMean() finds it
    # val_data_transforms.append(mytransforms.ConvertSegmentation())
    # val_data_transforms.append(mytransforms.ToTensor())
    # val_data_transforms.append(mytransforms.Relabel(255, -100))  # -100 is PyTorch's default ignore label
    # val_data_transforms.append(mytransforms.NormalizeZeroMean())
    # # val_data_transforms.append(mytransforms.Resize((150, 500), image_types=['color']))
    # val_dataset = StandardDataset(
    #     dataset=dataset,
    #     trainvaltest_split='validation',
    #     labels=labels,
    #     keys_to_load=keys_to_load,
    #     data_transforms=val_data_transforms,
    #     simple_mode=False
    # )
    # if 'sequence' in dataset:
    #     subset_dict = {}
    #     for k, v in val_dataset.data.items():
    #         subset_list = []
    #         for datapoint in val_dataset.data[k]:
    #             if 'val/lindau' in datapoint:
    #                 subset_list.append(datapoint)
    #         subset_dict[k] = subset_list
        
    #     val_dataset.data = subset_dict

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=False
    # )
    # metric = SegmentationRunningScore(19)
    # iterate(val_loader, model, metric, keys_to_load=keys_to_load)
