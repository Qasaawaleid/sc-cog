import os
import torch
from .helpers import make_scheduler
from .constants import SD_MODELS
from cog import Path
import cv2


def generate(
    prompt,
    negative_prompt,
    width,
    height,
    num_outputs,
    num_inference_steps,
    guidance_scale,
    scheduler,
    seed,
    output_image_ext,
    model,
    txt2img_pipe
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    extra_kwargs = {}

    prompt_prefix = SD_MODELS[model].get("prompt_prefix", None)
    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"

    negative_prompt_prefix = SD_MODELS[model].get(
        "negative_prompt_prefix", None)
    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"

    print(f"-- Prompt: {prompt} --")
    print(f"-- Negative Prompt: {negative_prompt} --")

    pipe = txt2img_pipe
    pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)
    generator = torch.Generator("cuda").manual_seed(seed)
    output = pipe(
        prompt=[prompt] * num_outputs if prompt is not None else None,
        negative_prompt=[negative_prompt] *
        num_outputs if negative_prompt is not None else None,
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
            cv2.imwrite(output_path_jpg, pngMat, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 90])
            output_path = output_path_jpg
        output_paths.append(Path(output_path))
    return output_paths
