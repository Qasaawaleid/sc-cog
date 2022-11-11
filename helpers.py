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

from constants import LANG_TO_ID, MODEL_CACHE
from lingua import Language


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


eng_score_max = 0.9
target_lang = Language.ENGLISH
target_lang_id = LANG_TO_ID[target_lang.name]

def translate_text(text, model, tokenizer, detector, label):
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""
    startTimeTranslation = time.time()
    translated_text = ""
    text_lang_id = target_lang_id
    
    confidence_values = detector.compute_language_confidence_values(text)
    eng_value = None
    detected_lang = None
    detected_lang_score = None
    
    for index in range(len(confidence_values)):
        curr = confidence_values[index]
        if index == 0:
            detected_lang = curr[0]
            detected_lang_score = curr[1]
        if curr[0] == Language.ENGLISH:
            eng_value = curr[1]
            
    if detected_lang != target_lang and (eng_value is None or eng_value < eng_score_max) and LANG_TO_ID.get(detected_lang.name) is not None:
        text_lang_id = LANG_TO_ID[detected_lang.name]
    
    if detected_lang is not None:
        print(f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --')
    print(f'-- {label} - Selected text language id: "{text_lang_id}" --')
    
    if text_lang_id != target_lang_id:
        translate = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang=text_lang_id,
            tgt_lang=target_lang_id,
            device=0
        )
        translate_output = translate(text, max_length=500)
        translated_text = translate_output[0]['translation_text']
        print(f'-- {label} - Translated text is: "{translated_text}" --')
    else:
        translated_text = text
        print(f"-- {label} - Text is already in the correct language, no translation needed --")
    
    endTimeTranslation = time.time()
    print(f"-- {label} - Translation done in: {endTimeTranslation - startTimeTranslation} sec. --")
    
    return translated_text
