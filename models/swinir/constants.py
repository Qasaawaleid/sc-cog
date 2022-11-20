import os

TASKS_SWINIR = {
  'Real-World Image Super-Resolution-Medium': 'real_sr',
  'Real-World Image Super-Resolution-Large': 'real_sr',
  'Grayscale Image Denoising': 'gray_dn',
  'Color Image Denoising': 'color_dn',
  'JPEG Compression Artifact Reduction': 'jpeg_car'
}

MODEL_DIR_SWINIR = 'experiments/pretrained_models'

MODELS_SWINIR = {
  'real_sr': {
    "medium": os.path.join(MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"),
    "large": os.path.join(MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"),
  },
  'gray_dn': {
      15: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
      25: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
      50: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth')
  },
  'color_dn': {
      15: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
      25: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
      50: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth')
  },
  'jpeg_car': {
      10: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth'),
      20: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth'),
      30: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth'),
      40: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')
  }
}