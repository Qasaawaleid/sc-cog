from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE
import concurrent.futures
from diffusers import (
    StableDiffusionPipeline,
)


def download_sd_model(key):
    model_id = SD_MODELS[key]['id']
    print(f"⏳ Downloading model: {model_id}")
    StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=SD_MODEL_CACHE
    )
    print(f"✅ Downloaded model: {key}")
    return {
        "key": key
    }


def download_sd_models_concurrently():
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [executor.submit(
            download_sd_model, key) for key in SD_MODELS
        ]
        # Wait for all tasks to complete
        results = [task.result()
                   for task in concurrent.futures.as_completed(download_tasks)]
    executor.shutdown(wait=True)


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
