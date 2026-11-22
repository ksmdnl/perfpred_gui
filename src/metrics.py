import numpy as np

COEFF = np.array([0.09334191 -0.6773528  -9.46617536])
COEFF = COEFF[-1:]


def get_psnr(img, gen_img):
    img = np.asarray(img, dtype=np.float32)
    gen_img = np.asarray(gen_img, dtype=np.float32)

    mse = np.mean((img - gen_img) ** 2)

    if mse < 1e-12:
        return float("inf")

    max_pixel = 255.0 if img.max() > 1.0 else 1.0

    psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return float(psnr)

def get_miou_estimate(psnr):
    return 0.09334191 * (psnr**2) - 0.6773528 * psnr + 9.46617536