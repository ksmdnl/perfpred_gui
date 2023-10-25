#!/usr/bin/env python
import os
import cv2
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .attack import attack
from dataloader.definitions.labels_file import *

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

reg_coeff = np.array([0.09334191 -0.6773528  -9.46617536])

# TODO: make an object of this
# so that it can become more flexible
def load_model(model_name="SwiftNetRec",
    backbone="resnet18",
    rec_decoder="swiftnet",
    num_class=19,                           # Cityscapes
    load_seg=False,
    device=torch.device('cpu'),
    dataset='cityscapes',
):
    config = {
        'name': model_name,
        'backbone': backbone,
        'decoder': rec_decoder,
        'trained': load_seg,
        'num_classes': num_class,
        'dataset': dataset,
    }
    print("Model is loaded with following parameters:")

    # TODO: load_model_def using config dictionary
    model = load_model_def(model_name, num_class, rec_decoder=rec_decoder)
    if load_seg:
        weightspath = os.path.join("builder/weights", model_name.lower(), backbone, 'model.pth')
        assert os.path.exists(weightspath), f"{weightspath} does not exists."
        model.load_state_dict(torch.load(weightspath, map_location=device))
        print("\nSegmentation weights are sucessfully loaded!")

    model = model.to(device)
    model.eval()
    return model

def get_miou_estimate(psnr):
    return 0.09334191 * (psnr**2) - 0.6773528 * psnr - 9.46617536

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

def generate_attack(epsilon, model, attack_type="fgsm", iterations=40):
    return attack.Attack(epsilon, model, type=attack_type, iterations=iterations)

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

def play_images_as_video(directory):
    # Get the list of image filenames in the directory and sort them
    images = sorted(os.listdir(directory))

    # Read the first image to get dimensions
    first_image_path = os.path.join(directory, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    output_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    # Iterate over the images, read each image, and write it to the video
    for image_name in images:
        image_path = os.path.join(directory, image_name)
        image = cv2.imread(image_path)
        output_video.write(image)
        cv2.imshow('Video Playback', image)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Release the VideoWriter and close any open windows
    output_video.release()
    cv2.destroyAllWindows()

def inference(model, path="builder/videos/tbilisi.mov"):
    cap = cv2.VideoCapture(path)
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
            if output[1] is not None:
                reconstructed = output[1]
                input_tensor = denormalize(input_tensor)
                psnr = get_psnr(input_tensor, reconstructed)
                print(f"PSNR: {psnr} dB")
                print(f"mIoU estimate: {get_miou_estimate(psnr)} %")
                cv2.imshow("Reconstruction", reconstructed)
                cv2.moveWindow("Reconstruction", 200, 200)

            cv2.imshow("Ground truth", frame)
            cv2.moveWindow("Segmentation", 400, 400)
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


class Evaluator(object):
    # CONF MATRIX
    #     0  1  2  (PRED)
    #  0 |TP FN FN|
    #  1 |FP TP FN|
    #  2 |FP FP TP|
    # (GT)
    # -> rows (axis=1) are FN
    # -> columns (axis=0) are FP
    @staticmethod
    def iou(conf):  # TP / (TP + FN + FP)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
        meaniu = np.nanmean(iu)
        result = {'iou': dict(zip(range(len(iu)), iu)), 'meaniou': meaniu}
        return result

    @staticmethod
    def accuracy(conf):  # TP / (TP + FN) aka 'Recall'
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            totalacc = np.diag(conf).sum() / (conf.sum())
            acc = np.diag(conf) / (conf.sum(axis=1))
        meanacc = np.nanmean(acc)
        result = {'totalacc': totalacc, 'meanacc': meanacc, 'acc': acc}
        return result

    @staticmethod
    def precision(conf):  # TP / (TP + FP)
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            prec = np.diag(conf) / (conf.sum(axis=0))
        meanprec = np.nanmean(prec)
        result = {'meanprec': meanprec, 'prec': prec}
        return result

    @staticmethod
    def freqwacc(conf):
        # Add 'add' in order to avoid division by zero and consequently NaNs in iu
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            iu = np.diag(conf) / (conf.sum(axis=1) + conf.sum(axis=0) - np.diag(conf))
            freq = conf.sum(axis=1) / (conf.sum())
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        result = {'freqwacc': fwavacc}
        return result

    @staticmethod
    def depththresh(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        result = {'delta1': a1, 'delta2': a2, 'delta3': a3}
        return result

    @staticmethod
    def deptherror(gt, pred):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log}
        return result

class SegmentationRunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask_true = (label_true >= 0) & (label_true < n_class)
        mask_pred = (label_pred >= 0) & (label_pred < n_class)
        mask = mask_pred & mask_true
        label_true = label_true[mask].astype(int)
        label_pred = label_pred[mask].astype(int)
        hist = np.bincount(n_class * label_true + label_pred,
                           minlength=n_class*n_class).reshape(n_class, n_class).astype(float)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, listofparams=None):
        """Returns the evaluation params specified in the list"""
        possibleparams = {
            'iou': Evaluator.iou,
            'acc': Evaluator.accuracy,
            'freqwacc': Evaluator.freqwacc,
            'prec': Evaluator.precision
        }
        if listofparams is None:
            listofparams = possibleparams

        result = {}
        for param in listofparams:
            if param in possibleparams.keys():
                result.update(possibleparams[param](self.confusion_matrix))
        return result

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
