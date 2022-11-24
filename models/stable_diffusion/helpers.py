from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from .constants import SD_MODEL_CACHE, SD_MODEL_ID


def make_scheduler(name):
    return {
        "PNDM": PNDMScheduler.from_config(
            SD_MODEL_ID,
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler"
        ),
        "K_LMS": LMSDiscreteScheduler.from_config(
            SD_MODEL_ID,
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler"
        ),
        "DDIM": DDIMScheduler.from_config(
            SD_MODEL_ID,
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler"
        ),
        "K_EULER": EulerDiscreteScheduler.from_config(
            SD_MODEL_ID,
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler"
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(
            SD_MODEL_ID,
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True,
            subfolder="scheduler"
        ),
    }[name]