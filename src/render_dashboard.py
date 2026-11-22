import pickle

import cv2
import numpy as np

from src.dasboard import labeled_image,create_status_panel,create_monitor_plot

HEIGHT = 180
WIDTH = 390

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 320

MAX_MIOU = 53

def render_dashboard(
    results_file="results.pkl",
    output_video="dashboard.mp4",
    fps=30,
):
    """
    Render dashboard video from stored inference results.
    """

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # dashboard_width = WIDTH * 2
    # dashboard_height = HEIGHT * 2 + 300
    dashboard_width = DISPLAY_WIDTH * 2
    dashboard_height = DISPLAY_HEIGHT * 2 + 400

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            dashboard_width,
            dashboard_height,
        ),
    )

    frame_history = []
    psnr_history = []
    miou_history = []

    n_frames = len(results["frames"])

    print(f"Rendering {n_frames} frames...")

    for idx in range(n_frames):

        frame = results["frames"][idx]
        segmentation = results["segmentations"][idx]
        reconstruction = results["reconstructions"][idx]

        status = results["status"][idx]
        metrics = results["metrics"][idx]

        psnr = metrics["psnr"]
        miou = metrics["estimated_miou"] / MAX_MIOU * 100

        frame_history.append(idx)
        psnr_history.append(psnr)
        miou_history.append(miou)

        camera_view = labeled_image(
            frame,
            "Camera",
        )

        reconstruction_view = labeled_image(
            reconstruction,
            "Reconstruction",
        )

        segmentation_view = labeled_image(
            segmentation,
            "Segmentation",
        )

        status_view = create_status_panel(
            status
        )

        # Normalize everything to dashboard resolution
        camera_view = cv2.resize(
            camera_view,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        reconstruction_view = cv2.resize(
            reconstruction_view,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        segmentation_view = cv2.resize(
            segmentation_view,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        status_view = cv2.resize(
            status_view,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        top_row = np.hstack([
            camera_view,
            reconstruction_view,
        ])

        middle_row = np.hstack([
            segmentation_view,
            status_view,
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
                400,
            ),
        )

        dashboard = np.vstack(
            [
                top_row,
                middle_row,
                plot_img,
            ]
        )

        #
        # Portfolio header
        #
        cv2.putText(
            dashboard,
            "Real-Time Perception Quality Monitoring",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        #
        # Frame counter
        #
        cv2.putText(
            dashboard,
            f"Frame {idx + 1}/{n_frames}",
            (
                dashboard.shape[1] - 220,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(dashboard)

        cv2.imshow(
            "Autonomous Driving Perception Dashboard",
            dashboard,
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyAllWindows()

    print(
        f"Dashboard video written to '{output_video}'"
    )


if __name__ == "__main__":
    render_dashboard()