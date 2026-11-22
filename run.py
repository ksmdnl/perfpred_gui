import os
import sys

# from src.inference import run_inference
from src.inference_compr import run_inference
from src.collect_predictions import collect_predictions
from src.model import load_model

DEVICE = os.getenv("DEVICE", "cpu")

def main(sample_path: str, model_name: str, domain: str) -> None:
    weights_path = os.path.join('weights', model_name)
    model = load_model(
        weights_path=weights_path,
        device=DEVICE,
        domain=domain
    )

    if sys.argv[1] == 'inference':
        run_inference(model, path=sample_path)
    elif sys.argv[1] == 'collect':
        collect_predictions(model, sample_path)
    else:
        raise ValueError(f"Please specify runtime.")
    return 0

if __name__ == "__main__":
    # Path to a sample video
    sample_path = "./samples/stuttgart_00.mp4"

    # Domain class is Cityscape as default: 19 classes
    domain = "cityscapes"

    # <SEGMENTATION_ARCH>_<BACKBONE_ARCH>
    model_name = "swiftnetrec_rn50"

    main(sample_path, model_name, domain)
