from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from .constants import SD_MODEL_CACHE, SD_MODELS


def make_scheduler(name, model, revision):
    return {
        "PNDM": PNDMScheduler.from_config(
            SD_MODELS[model]["id"],
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True, 
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_LMS": LMSDiscreteScheduler.from_config(
            SD_MODELS[model]["id"],
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "DDIM": DDIMScheduler.from_config(
            SD_MODELS[model]["id"],
            cache_dir=SD_MODEL_CACHE,
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_EULER": EulerDiscreteScheduler.from_config(
            SD_MODELS[model]["id"],
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True, 
            subfolder="scheduler",
            revision=revision or "main"
        ),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(
            SD_MODELS[model]["id"],
            cache_dir=SD_MODEL_CACHE, 
            local_files_only=True,
            subfolder="scheduler",
            revision=revision or "main"
        ),
    }[name]