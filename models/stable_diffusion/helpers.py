from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
)


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def download_sd_model(key):
    print(f"⏳ Downloading model: {key}")
    StableDiffusionPipeline.from_pretrained(
        SD_MODELS[key]["id"],
        cache_dir=SD_MODEL_CACHE
    )
    print(f"✅ Downloaded model: {key}")
    return {
        "key": key
    }
