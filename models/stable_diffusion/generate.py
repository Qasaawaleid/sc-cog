import os
import torch
from PIL import Image
from .helpers import make_scheduler
from cog import Path
import cv2

def generate(
  prompt,
  negative_prompt,
  width,
  height,
  init_image,
  mask,
  prompt_strength,
  num_outputs,
  num_inference_steps,
  guidance_scale,
  scheduler,
  seed,
  output_image_ext,
  model,
  txt2img,
  img2img,
  inpaint,
  txt2img_oj
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    """ if width * height > 786432:
        raise ValueError(
            "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        ) """

    extra_kwargs = {}
    if mask:
        if not init_image:
            raise ValueError("mask was provided without init_image")
        pipe = inpaint
        init_image = Image.open(init_image).convert("RGB")
        extra_kwargs = {
            "mask_image": Image.open(mask).convert("RGB").resize(init_image.size),
            "init_image": init_image,
            "strength": prompt_strength,
        }
    elif init_image:
        pipe = img2img
        extra_kwargs = {
            "init_image": Image.open(init_image).convert("RGB"),
            "strength": prompt_strength,
        }
    elif model == "Openjourney":
        pipe = txt2img_oj
    else:
        pipe = txt2img

    pipe.scheduler = make_scheduler(scheduler)
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt=[prompt] * num_outputs if prompt is not None else None,
        negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
        **extra_kwargs,
    )

    samples_filtered = [
        output.images[i]
        for i, nsfw_flag in enumerate(output.nsfw_content_detected)
        if not nsfw_flag
    ]

    """ if len(samples_filtered) == 0:
        raise Exception(
            f"NSFW content detected. Try running it again, or try a different prompt."
        ) """

    if num_outputs > len(samples_filtered):
        print(
            f"NSFW content detected in {num_outputs - len(samples_filtered)} outputs, showing the rest {len(samples_filtered)} images..."
        )

    samples = output.images
    output_paths = []
    for i, sample in enumerate(samples):
        output_path = f"/tmp/out-{i}.png"
        sample.save(output_path)
        if output_image_ext == "jpg":
            output_path_jpg = f"/tmp/out-{i}.jpg"
            pngMat = cv2.imread(output_path)
            cv2.imwrite(output_path_jpg, pngMat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            output_path = output_path_jpg
        output_paths.append(Path(output_path))
    return output_paths