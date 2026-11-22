import pickle
import cv2
import numpy as np
import torch

from src.processing import preprocess, postprocess, denormalize
from src.metrics import get_psnr, get_miou_estimate
from src.monitor import PerceptionMonitor


def collect_predictions(model, video_path):
    cap = cv2.VideoCapture(video_path)

    monitor = PerceptionMonitor()

    results = {
        "metadata": {
            "video_path": video_path,
        },

        "frames": [],
        "segmentations": [],
        "reconstructions": [],

        "metrics": [],

        "status": [],
        "psnr": [],
        "miou": [],
    }

    frame_idx = 0

    max_miou = -1

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        input_tensor = preprocess(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        mask, reconstruction = postprocess(output)

        if reconstruction is None:
            continue

        input_denorm = denormalize(input_tensor)

        psnr = get_psnr(
            input_denorm,
            reconstruction,
        )

        miou = np.clip(
            get_miou_estimate(psnr),
            0,
            100,
        ) / 55 * 100

        if miou > max_miou:
            max_miou = miou

        status = monitor.update(
            psnr=psnr,
            miou=miou,
        )

        results["frames"].append(frame)
        results["segmentations"].append(mask)
        results["reconstructions"].append(reconstruction)

        results["psnr"].append(float(psnr))
        results["miou"].append(float(miou))

        results["metrics"].append({
            "frame_idx": frame_idx,
            "psnr": float(psnr),
            "estimated_miou": float(miou),
        })

        results["status"].append(status)

        frame_idx += 1

        if frame_idx % 5 == 0:
            print(
                f"Processed {frame_idx} frames..."
            )

    cap.release()
    print(f"MAX MIOU: {max_miou}")

    with open(
        "results.pkl",
        "wb",
    ) as f:
        pickle.dump(
            results,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved {frame_idx} frames to results.pkl"
    )