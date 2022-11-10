import os
import shutil
import time
import torch
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
from gfpgan import GFPGANer
from transformers import pipeline

from constants import LOCALE_TO_ID, MODEL_CACHE


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def make_scheduler(name):
    return {
        "PNDM": PNDMScheduler.from_config(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler"
        ),
        "K-LMS": LMSDiscreteScheduler.from_config(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler"
        ),
        "DDIM": DDIMScheduler.from_config(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler"
        ),
        "K_EULER": EulerDiscreteScheduler.from_config(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler"
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE, 
            local_files_only=True,
            subfolder="scheduler"
        ),
    }[name]

  
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


score_min = 0.6
target_lang_id = LOCALE_TO_ID["en"]

def translate_text(text, model, tokenizer, detector):
    if text == "":
        print("-- No text to translate, skipping")
        return ""
    startTimeTranslation = time.time()
    translated_text = ""
    text_locale_res = detector(text)[0]
    text_locale = text_locale_res["label"]
    text_locale_score = text_locale_res["score"]
    text_locale_id = LOCALE_TO_ID["en"]
    if LOCALE_TO_ID.get(text_locale) is not None and text_locale_score > score_min:
        text_locale_id = LOCALE_TO_ID[text_locale]
    
    print(f'-- Guessed text locale: "{text_locale}". Score: {text_locale_score} --')
    print(f"-- Selected text locale id: {text_locale_id} --")
    
    if text_locale_id != target_lang_id:
        translate = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang=text_locale_id,
            tgt_lang=target_lang_id,
            device=0
        )
        translate_output = translate(text, max_length=500)
        translated_text = translate_output[0]['translation_text']
        print(f'-- Translated text is: "{translated_text}"')
    else:
        translated_text = text
        print(f"-- Text is already in the correct language, no translation needed")
    
    endTimeTranslation = time.time()
    print(f"-- Translation done in: {endTimeTranslation - startTimeTranslation} sec. --")
    
    return translated_text
