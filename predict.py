import os
from typing import Optional, List

import torch
from torch import autocast
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image
from cog import BasePredictor, Input, Path

from text_to_image import (
    StableDiffusionPipeline
)

import cv2
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from gfpgan import GFPGANer
from helpers import clean_folder
import time


MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        scheduler = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        
        # For the upscaler
        os.makedirs('output', exist_ok=True)
        # download weights
        if not os.path.exists('weights/realesr-general-x4v3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./weights'
            )
        if not os.path.exists('weights/GFPGANv1.4.pth'):
            os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./weights')
        if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights'
            )
        if not os.path.exists('weights/RealESRGAN_x4plus_anime_6B.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P ./weights'
            )
        if not os.path.exists('weights/realesr-animevideov3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -P ./weights'
            )
    
    def choose_model(self, scale, version, tile=0):
        half = True if torch.cuda.is_available() else False
        if version == 'General - RealESRGANplus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'weights/RealESRGAN_x4plus.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'General - v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = 'weights/realesr-general-x4v3.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'Anime - anime6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = 'weights/RealESRGAN_x4plus_anime_6B.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'AnimeVideo - v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            model_path = 'weights/realesr-animevideov3.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)

        self.face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.4.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        negative_prompt: str = Input(description="Negative prompt", default=None),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        img: Path = Input(description='Input', default=None),
        version: str = Input(
            description='RealESRGAN version. Please see [Readme] below for more descriptions',
            choices=['General - RealESRGANplus', 'General - v3', 'Anime - anime6B', 'AnimeVideo - v3'],
            default='General - v3'),
        scale: float = Input(description='Rescaling factor', default=2),
        face_enhance: bool = Input(
            description='Enhance faces with GFPGAN. Note that it does not work for anime images/vidoes', default=False),
        tile: int = Input(
            description=
            'Tile size. Default is 0, that is no tile. When encountering the out-of-GPU-memory issue, please specify it, e.g., 400 or 200',
            default=0),
        process_type: str = Input(
            description=
            'Process type. It can be generate or upscale',
            choices=['generate', 'upscale'],
            default='generate')
    ) -> List[Path]:
        if process_type == 'upscale':
            startTime = time.time()
            if img is None:
                raise Exception("Selected mode is upscale, an image is required")
            if tile <= 100 or tile is None:
                tile = 0
            print(f'img: {img}. version: {version}. scale: {scale}. face_enhance: {face_enhance}. tile: {tile}.')
            try:
                extension = os.path.splitext(os.path.basename(str(img)))[1]
                img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img_mode = 'RGBA'
                elif len(img.shape) == 2:
                    img_mode = None
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img_mode = None

                h, w = img.shape[0:2]
                if h < 300:
                    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

                self.choose_model(scale, version, tile)

                try:
                    if face_enhance:
                        _, _, output = self.face_enhancer.enhance(
                            img, has_aligned=False, only_center_face=False, paste_back=True)
                    else:
                        output, _ = self.upsampler.enhance(img, outscale=scale)
                except RuntimeError as error:
                    print('Error', error)
                    print('If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.')

                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                # save_path = f'output/out.{extension}'
                # cv2.imwrite(save_path, output)
                
                # out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
                # force jpg for smaller size
                out_path = Path(tempfile.mkdtemp()) / f'out.jpg'
                cv2.imwrite(str(out_path), output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            except Exception as error:
                print('global exception: ', error)
            finally:
                clean_folder('output')
            output_paths = []
            output_paths.append(out_path)
            endTime = time.time()
            print(f"-- Upscaled in: {endTime - startTime} sec. --")
            return output_paths
        else:
            startTime = time.time()
            """Run a single prediction on the model"""
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")

            if width == height == 1024:
                raise ValueError(
                    "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
                )

            # use LMS without init images
            scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )

            self.pipe.scheduler = scheduler

            generator = torch.Generator("cuda").manual_seed(seed)
            output = self.pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs if negative_prompt is not None else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
            """ if any(output["nsfw_content_detected"]):
                raise Exception("NSFW content detected, please try a different prompt") """

            output_paths = []
            for i, sample in enumerate(output["sample"]):
                output_path = f"/tmp/out-{i}.png"
                sample.save(output_path)
                output_paths.append(Path(output_path))
            endTime = time.time()
            print(f"-- Generated in: {endTime - startTime} sec. --")
            return output_paths