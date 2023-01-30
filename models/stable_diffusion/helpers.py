from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
)
import concurrent.futures


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


def download_sd_models_concurrently(keys):
    with concurrent.futures.ThreadPoolExecutor(6) as executor:
        # Start the download tasks
        download_tasks = [executor.submit(
            download_sd_model, key) for key in keys]
        # Wait for all tasks to complete
        results = [task.result()
                   for task in concurrent.futures.as_completed(download_tasks)]
