import cv2
import numpy as np
import matplotlib.pyplot as plt


factor = 1
# HEIGHT, WIDTH = int(180 * factor), int(390 * factor)
HEIGHT, WIDTH = int(360 * factor), int(750 * factor)

def create_monitor_plot(
    frame_history,
    miou_history,
    psnr_history,
):
    fig, ax1 = plt.subplots(figsize=(8, 3))

    ax1.plot(
        frame_history,
        miou_history,
        color="tab:blue",
        label="Estimated mIoU"
    )

    ax1.set_ylabel("mIoU [%]")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()

    ax2.plot(
        frame_history,
        psnr_history,
        color="tab:red",
        label="PSNR"
    )

    ax2.set_ylabel("PSNR [dB]")

    ax1.grid(True)

    fig.tight_layout()

    fig.canvas.draw()

    img = np.frombuffer(
        fig.canvas.buffer_rgba(),
        dtype=np.uint8
    )

    img = img.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGBA2BGR
    )

    plt.close(fig)

    return img

def create_status_panel(status):
    panel = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    state = status["state"].value

    colors = {
        "NOMINAL": (0, 200, 0),
        "CAUTION": (0, 200, 255),
        "DEGRADED": (0, 120, 255),
        "UNSAFE": (0, 0, 255),
    }

    color = colors[state]

    cv2.putText(
        panel,
        "Perception Monitor",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        panel,
        f"State: {state}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    cv2.putText(
        panel,
        f"OOD: {status['ood']}",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    cv2.putText(
        panel,
        f"mIoU: {status['miou']:.1f}%",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    cv2.putText(
        panel,
        f"Confidence: {status['confidence']:.2f}",
        (30, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    cv2.putText(
        panel,
        f"PSNR: {status['psnr']:.2f}",
        (30, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    return panel

def labeled_image(img, title):

    img = img.copy()

    cv2.rectangle(
        img,
        (0, 0),
        (img.shape[1], 40),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        img,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2,
    )

    return img