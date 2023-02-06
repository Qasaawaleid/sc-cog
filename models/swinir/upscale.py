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
from .helpers import define_model, get_image_pair, setup, get_args_swinir
import time

device = torch.device("cuda")


@torch.cuda.amp.autocast()
def upscale(image):
    if image is None:
        raise ValueError("Image is required for the upscaler.")

    args = get_args_swinir()
    args.task = TASKS_SWINIR["Real-World Image Super-Resolution-Large"]
    args.scale = 4
    args.model_path = MODELS_SWINIR["real_sr"]["large"]
    args.large_model = True

    output_image = None
    try:
        start_time = time.time()
        # set input folder
        input_dir = "input_cog_temp"
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, os.path.basename(image))
        shutil.copy(str(image), input_path)

        args.folder_lq = input_dir

        start_time_load = time.time()
        model = define_model(args)
        model.eval()
        model = model.to(device)
        end_time_load = time.time()
        print(f"Load model time: {round((end_time_load - start_time_load) * 1000)}ms")

        # setup folder and path
        folder, save_dir, border, window_size = setup(args)
        os.makedirs(save_dir, exist_ok=True)
        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []
        test_results["psnr_b"] = []
        # psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0
        end_time = time.time()
        print(f"Setup time: {round((end_time - start_time) * 1000)}ms")

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, "*")))):
            start_time = time.time()
            # read image
            imgname, img_lq, img_gt = get_image_pair(
                args, path
            )  # image to HWC-BGR, float32
            img_lq = np.transpose(
                img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
            )  # HCW-BGR to CHW-RGB
            img_lq = (
                torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
            )  # CHW-RGB to NCHW-RGB
            end_time = time.time()
            print(f"Read image time: {round((end_time - start_time) * 1000)}ms")

            # inference
            start_time = time.time()
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                    :, :, : h_old + h_pad, :
                ]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                    :, :, :, : w_old + w_pad
                ]
                output = model(img_lq)
                output = output[..., : h_old * args.scale, : w_old * args.scale]
            end_time = time.time()
            print(f"Inference time: {round((end_time - start_time) * 1000)}ms - {idx}")

            start_time = time.time()
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            # float32 to uint8
            output = (output * 255.0).round().astype(np.uint8)
            output_image = output
            end_time = time.time()
            print(f"Save image time: {round((end_time - start_time) * 1000)}ms")
    finally:
        start_time = time.time()
        clean_folder(input_dir)
        end_time = time.time()
        print(f"Cleanup time: {round((end_time - start_time) * 1000)}ms")
    return output_image
