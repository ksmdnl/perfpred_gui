from collections import deque
from enum import Enum

import numpy as np


class PerceptionState(Enum):
    NOMINAL = "NOMINAL"
    CAUTION = "CAUTION"
    DEGRADED = "DEGRADED"
    UNSAFE = "UNSAFE"


class PerceptionMonitor:
    """
    Runtime monitor for segmentation quality.

    Inputs:
        PSNR
        estimated mIoU

    Outputs:
        smoothed mIoU
        confidence score
        safety state
        OOD detection flag
    """

    def __init__(
        self,
        window_size=30,
        nominal_threshold=90.0,
        caution_threshold=80.0,
        degraded_threshold=70.0,
        sudden_drop_threshold=15.0,
        min_psnr=20.0,
    ):
        self.window_size = window_size

        self.nominal_threshold = nominal_threshold
        self.caution_threshold = caution_threshold
        self.degraded_threshold = degraded_threshold

        self.sudden_drop_threshold = sudden_drop_threshold
        self.min_psnr = min_psnr

        self.miou_history = deque(maxlen=window_size)
        self.psnr_history = deque(maxlen=window_size)

        self.prev_miou = None

    def _get_state(self, smoothed_miou):

        if smoothed_miou >= self.nominal_threshold:
            return PerceptionState.NOMINAL

        if smoothed_miou >= self.caution_threshold:
            return PerceptionState.CAUTION

        if smoothed_miou >= self.degraded_threshold:
            return PerceptionState.DEGRADED

        return PerceptionState.UNSAFE

    def _is_ood(
        self,
        psnr,
        current_miou,
        smoothed_miou,
        miou_drop,
    ):
        """
        OOD criteria:
            1. low reconstruction quality
            2. low segmentation quality
            3. sudden segmentation collapse
        """

        low_psnr = psnr < self.min_psnr

        low_miou = current_miou < self.degraded_threshold

        sudden_drop = miou_drop > self.sudden_drop_threshold

        return (
            low_psnr
            or low_miou
            or sudden_drop
        )

    def update(self, psnr, miou):

        self.psnr_history.append(psnr)
        self.miou_history.append(miou)

        smoothed_miou = float(np.mean(self.miou_history))
        smoothed_psnr = float(np.mean(self.psnr_history))

        confidence = np.clip(
            smoothed_miou / 100.0,
            0.0,
            1.0
        )

        if self.prev_miou is None:
            miou_drop = 0.0
        else:
            miou_drop = max(
                0.0,
                self.prev_miou - miou
            )

        self.prev_miou = miou

        state = self._get_state(smoothed_miou)

        ood = self._is_ood(
            psnr=psnr,
            current_miou=miou,
            smoothed_miou=smoothed_miou,
            miou_drop=miou_drop,
        )

        return {
            "psnr": float(psnr),
            "smoothed_psnr": smoothed_psnr,
            "miou": float(miou),
            "smoothed_miou": smoothed_miou,
            "miou_drop": miou_drop,
            "confidence": float(confidence),
            "ood": bool(ood),
            "state": state,
        }