import cv2
import warnings
import importlib
import random

import numpy as np

import torch

from src.monitor import PerceptionMonitor
from src.dasboard import labeled_image, create_status_panel, create_monitor_plot
from src.metrics import get_psnr, get_miou_estimate
from src.processing import preprocess, postprocess, denormalize

from robustness.attacks import ATTACKS
from dataloader.definitions.labels_file import dataset_labels

trainid2label = dataset_labels['cityscapes'].gettrainid2label()

MAX_MIOU = 50

# do_resize = os.getenv("RESIZE", True)
do_resize = False
factor = 1
HEIGHT, WIDTH = int(360 * factor), int(750 * factor)
# HEIGHT, WIDTH = 300, 1000

monitor = PerceptionMonitor()

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
MAX_HISTORY = 200

def run_inference(model, path="builder/videos/tbilisi.mov"):
    cap = cv2.VideoCapture(path)
    frame_idx = 0

    frame_history = []
    miou_history = []
    psnr_history = []

    monitor = PerceptionMonitor()

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
            frame_history.append(frame_idx)
            miou_history.append(miou)
            psnr_history.append(psnr)

            frame_history = frame_history[-MAX_HISTORY:]
            miou_history = miou_history[-MAX_HISTORY:]
            psnr_history = psnr_history[-MAX_HISTORY:]


            camera_view = labeled_image(
                frame,
                "Camera"
            )

            reconstruction_view = labeled_image(
                reconstructed,
                "Reconstruction"
            )

            segmentation_view = labeled_image(
                mask,
                "Segmentation"
            )

            status_view = create_status_panel(status)

            top_row = np.hstack([
                camera_view,
                reconstruction_view
            ])

            middle_row = np.hstack([
                segmentation_view,
                status_view
            ])

            plot_img = create_monitor_plot(
                frame_history,
                miou_history,
                psnr_history,
            )

            plot_img = cv2.resize(
                plot_img,
                (
                    top_row.shape[1],
                    300
                )
            )

            dashboard = np.vstack([
                top_row,
                middle_row,
                plot_img
            ])

        cv2.imshow(
            "Autonomous Driving Perception Dashboard",
            dashboard
        )
        frame_idx += 1
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
