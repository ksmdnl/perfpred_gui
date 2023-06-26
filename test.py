#!/usr/bin/env python
from builder.util import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "SwiftNetRec"
    num_classes = 19
    backbone = "resnet18"
    rec_decoder = "swiftnet"
    model = load_model_def(model_name, num_classes, rec_decoder=rec_decoder)
    weightspath = os.path.join("builder/weights", model_name.lower(), backbone, 'model.pth')
    assert os.path.exists(weightspath), f"{weightspath} does not exists."
    model.load_state_dict(torch.load(weightspath, map_location=device))

    model = model.to(device)
    model.eval()
    model = ModelWrapper(model, postprocessing=remove_rec_output)
    attack = generate_attack(10 / 255, model, attack_type="metzen", iterations=1)
    path = "builder/samples/munich_000068_000019_leftImg8bit.png"
    sample = single_frame(sample=path, inference=False)
    sample_og = denormalize(sample)