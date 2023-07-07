#!/usr/bin/env python
import cv2
from builder.util import *

def iterate(dataset, keys_to_load=['color', 'segmentation_trainid']):
    for (_, inputs) in enumerate(dataset):
        im = inputs[("color_aug", 0, 0)]
        targets = inputs[(keys_to_load[-1], 0, 0)][:, 0, :, :].long()
        print(targets.shape)
        exit(0)

if __name__ == "__main__":
    path = "builder/videos/output/output_video.avi"
    model = load_model(model_name="SwiftNetRec",
        backbone="resnet18",
        rec_decoder="swiftnet",
        num_class=19,
        load_seg=True,
        dataset=path
    )
    model = ModelWrapper(model, postprocessing=None)
    # with torch.no_grad():
    #     attack = generate_attack(10 / 255, model, attack_type="metzen", iterations=1)
    #     path = "builder/samples/munich_000068_000019_leftImg8bit.png"
    #     sample = single_frame(sample=path, inference=False)#.squeeze(0).permute(1,2,0).cpu().numpy()
    #     attacked_img, _ = attack.generate(sample)
    #     print(attacked_img.shape)
    #     attack_denormalize = denormalize(attacked_img)
    #     print(attack_denormalize.shape)
    #     cv2.imshow("attacked image denormalized", attacked_img.squeeze().permute((1,2,0)).cpu().numpy())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # inference(model, path=path)
    # exit(0)
    from dataloader.pt_data_loader.specialdatasets import StandardDataset
    import dataloader.pt_data_loader.mytransforms as mytransforms
    from torch.utils.data import DataLoader

    labels = labels_cityscape_seg.getlabels()  # original labels used by Cityscapes
    keys_to_load = ['color', 'segmentation_trainid']
    # keys_to_load = ['color']
    # subset = 'lindau'
    # dataset = 'cityscapes_sequence'
    subset = 'frankfurt'
    dataset = 'cityscapes'
    val_data_transforms = [mytransforms.CreateScaledImage()]
    val_data_transforms.append(mytransforms.CreateColoraug())  # Adjusts keys so that NormalizeZeroMean() finds it
    val_data_transforms.append(mytransforms.ConvertSegmentation())
    val_data_transforms.append(mytransforms.ToTensor())
    val_data_transforms.append(mytransforms.NormalizeZeroMean())
    val_dataset = StandardDataset(dataset=dataset,
                                    trainvaltest_split='validation',
                                    labels=labels,
                                    keys_to_load=keys_to_load,
                                    data_transforms=val_data_transforms,
                                    simple_mode=False)
    if dataset.split('/')[-1] == 'sequence':
        subset_dict = {}
        for k, v in val_dataset.data.items():
            subset_list = []
            for datapoint in val_dataset.data[k]:
                if 'val/lindau' in datapoint:
                    subset_list.append(datapoint)
            subset_dict[k] = subset_list
        
        val_dataset.data = subset_dict

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
    iterate(val_loader, keys_to_load=keys_to_load)