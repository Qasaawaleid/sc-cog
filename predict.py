import os
import time
from typing import List

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
)
from PIL import Image
from cog import BasePredictor, Input, Path
from helpers import make_scheduler, clean_folder, translate_text
import cv2
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from constants import MODEL_CACHE, TRANSLATOR_MODEL_CACHE, TRANSLATOR_TOKENIZER_CACHE
from lingua import LanguageDetectorBuilder

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Stable Diffusion v1.5 pipelines...")

        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        
        self.inpaint_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        
        # For translation
        self.detect_language = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
        
        translate_model_name = "facebook/nllb-200-distilled-1.3B"
        self.translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_name, cache_dir=TRANSLATOR_TOKENIZER_CACHE)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(translate_model_name, cache_dir=TRANSLATOR_MODEL_CACHE).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt.", default=""),
        negative_prompt: str = Input(description="Input negative prompt.", default=""),
        width: int = Input(
            description="Width of output image.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        init_image: Path = Input(
            description="Inital image to generate variations of. Will be resized to the specified width and height.",
            default=None,
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over init_image. Black pixels are inpainted and white pixels are preserved. Tends to work better with prompt strength of 0.5-0.7. Consider using https://replicate.com/andreasjansson/stable-diffusion-inpainting instead.",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image.",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
            ge=1,
            le=10,
            default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K-LMS",
            choices=["DDIM", "K-LMS", "PNDM", "K_EULER", "K_EULER_ANCESTRAL"],
            description="Choose a scheduler. If you use an init image, PNDM will be used.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.", default=None
        ),
        process_type: str = Input(
            description="Choose a process type, Can be 'generate' or 'upscale'.",
            choices=["generate", "upscale"],
            default="generate",
        )
    ) -> List[Path]:
        """ out_path = Path(tempfile.mkdtemp()) / f'out.jpg'
        cv2.imwrite(str(out_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) """
        if process_type == 'upscale':
            
            endTime = time.time()
            print(f"-- Upscaled in: {endTime - startTime} sec. --")
            return output_paths
        else:
            """Run a single prediction on the model"""
            startTime = time.time()
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
                pipe = self.inpaint_pipe
                init_image = Image.open(init_image).convert("RGB")
                extra_kwargs = {
                    "mask_image": Image.open(mask).convert("RGB").resize(init_image.size),
                    "init_image": init_image,
                    "strength": prompt_strength,
                }
            elif init_image:
                pipe = self.img2img_pipe
                extra_kwargs = {
                    "init_image": Image.open(init_image).convert("RGB"),
                    "strength": prompt_strength,
                }
            else:
                pipe = self.txt2img_pipe
                
            t_prompt = translate_text(
                prompt,
                self.translate_model,
                self.translate_tokenizer,
                self.detect_language,
                "Prompt"
            )
            t_negative_prompt = translate_text(
                negative_prompt,
                self.translate_model,
                self.translate_tokenizer,
                self.detect_language,
                "Negative prompt"
            )
            
            pipe.scheduler = make_scheduler(scheduler)
            generator = torch.Generator("cuda").manual_seed(seed)
            output = pipe(
                prompt=[t_prompt] * num_outputs if t_prompt is not None else None,
                negative_prompt=[t_negative_prompt] * num_outputs if t_negative_prompt is not None else None,
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
                output_paths.append(Path(output_path))
                
            endTime = time.time()
            print(f"-- Generated in: {endTime - startTime} sec. --")
            return output_paths