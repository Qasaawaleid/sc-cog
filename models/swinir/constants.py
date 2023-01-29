import os

MODEL_DIR_SWINIR = 'experiments/pretrained_models'

TASKS_SWINIR = {
    'Real-World Image Super-Resolution-Large': 'real_sr',
}
MODELS_SWINIR = {
    'real_sr': {
        "large": os.path.join(MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    }
}
