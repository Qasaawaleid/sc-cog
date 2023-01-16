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
    output_image_quality,
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

    output_paths = []
    nsfw_count = 0
    black_pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAACklEQVR4AWNkAAAABAACGr4IAwAAAABJRU5ErkJggg=="

    for i, nsfw_flag in enumerate(output.nsfw_content_detected):
        if nsfw_flag:
            nsfw_count += 1
            output_paths.append(Path(black_pixel))
        else:
            output_path = f"/tmp/out-{i}.png"
            output.images[i].save(output_path)
            if output_image_ext == "jpg" or output_image_ext == "webp":
                output_path_converted = f"/tmp/out-{i}.{output_image_ext}"
                pngMat = cv2.imread(output_path)
                quality_type = cv2.IMWRITE_JPEG_QUALITY
                if output_image_ext == "webp":
                    quality_type = cv2.IMWRITE_WEBP_QUALITY
                cv2.imwrite(
                    output_path_converted, pngMat,
                    [int(quality_type), output_image_quality]
                )
                output_path = output_path_converted
            output_paths.append(Path(output_path))

    if nsfw_count > 0:
        print(
            f"NSFW content detected in {nsfw_count}/{len(output_paths)} of the outputs."
        )

    return output_paths
