import glob
import os
import shutil
import tempfile
import torch
import shutil
import numpy as np
from collections import OrderedDict
import cv2
import tempfile
from common.helpers import clean_folder
from cog import Path
from .constants import MODELS_SWINIR, TASKS_SWINIR
from .helpers import define_model, get_image_pair, setup


def upscale(args, device, task, image, noise, jpeg):
  if image is None:
      raise ValueError("Image is required for the upscaler.")

  args.task = TASKS_SWINIR[task]
  args.noise = noise
  args.jpeg = jpeg

  if args.task == "real_sr":
      args.scale = 4
      if args.task == "Real-World Image Super-Resolution-Large":
          args.model_path = MODELS_SWINIR["real_sr"]["large"]
          args.large_model = True
      else:
          args.model_path = MODELS_SWINIR["real_sr"]["medium"]
          args.large_model = False
  elif args.task in ["gray_dn", "color_dn"]:
      args.model_path = MODELS_SWINIR[args.task][noise]
  else:
      args.model_path = MODELS_SWINIR[args.task][jpeg]

  try:
      # set input folder
      input_dir = 'input_cog_temp'
      os.makedirs(input_dir, exist_ok=True)
      input_path = os.path.join(input_dir, os.path.basename(image))
      shutil.copy(str(image), input_path)
      if args.task == 'real_sr':
          args.folder_lq = input_dir
      else:
          args.folder_gt = input_dir

      model = define_model(args)
      model.eval()
      model = model.to(device)

      # setup folder and path
      folder, save_dir, border, window_size = setup(args)
      os.makedirs(save_dir, exist_ok=True)
      test_results = OrderedDict()
      test_results['psnr'] = []
      test_results['ssim'] = []
      test_results['psnr_y'] = []
      test_results['ssim_y'] = []
      test_results['psnr_b'] = []
      # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0
      out_path = Path(tempfile.mkdtemp()) / "out.jpeg"

      for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
          # read image
          imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
          img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
          img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

          # inference
          with torch.no_grad():
              # pad input image to be a multiple of window_size
              _, _, h_old, w_old = img_lq.size()
              h_pad = (h_old // window_size + 1) * window_size - h_old
              w_pad = (w_old // window_size + 1) * window_size - w_old
              img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
              img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
              output = model(img_lq)
              output = output[..., :h_old * args.scale, :w_old * args.scale]

          # save image
          output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
          if output.ndim == 3:
              output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
          output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
          cv2.imwrite(str(out_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
  finally:
      clean_folder(input_dir)
  return [out_path]