import os
import torch
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler
)

SD_MODEL_CACHE = "diffusers-cache"

SD_MODELS = {}
models_from_env = os.environ.get(
    "MODELS", "Stable Diffusion v1.5,Openjourney")
models_from_env_list = models_from_env.split(",")
for model_env in models_from_env_list:
    if model_env == "Stable Diffusion v1.5":
        SD_MODELS[model_env] = {
            "id": "runwayml/stable-diffusion-v1-5",
            "revision": "fp16",
            "torch_dtype": torch.float16
        }
    elif model_env == "Openjourney":
        SD_MODELS[model_env] = {
            "id": "prompthero/openjourney",
            "prompt_prefix": "mdjrny-v4 style",
            "torch_dtype": torch.float16
        }
    elif model_env == "Redshift Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/redshift-diffusion",
            "prompt_prefix": "redshift style",
            "torch_dtype": torch.float16
        }
    elif model_env == "Arcane Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/Arcane-Diffusion",
            "prompt_prefix": "arcane style",
            "torch_dtype": torch.float16
        }
    elif model_env == "Ghibli Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/Ghibli-Diffusion",
            "prompt_prefix": "ghibli style",
            "torch_dtype": torch.float16
        }
    elif model_env == "Mo-Di Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/mo-di-diffusion",
            "prompt_prefix": "modern disney style",
            "torch_dtype": torch.float16
        }
    elif model_env == "Waifu Diffusion v1.4":
        SD_MODELS[model_env] = {
            "id": "hakurei/waifu-diffusion",
            "prompt_prefix": "masterpiece, best quality, high quality",
            "negative_prompt_prefix": "worst quality, low quality, deleted, nsfw",
            "torch_dtype": torch.float32
        }
    elif model_env == "22h Diffusion v0.1":
        SD_MODELS[model_env] = {
            "id": "22h/vintedois-diffusion-v0-1",
            "prompt_prefix": "estilovintedois",
            "torch_dtype": torch.float16
        }

SD_MODEL_CHOICES = list(SD_MODELS.keys())
SD_MODEL_DEFAULT = SD_MODEL_CHOICES[0]

SD_SCHEDULERS = {
    "K_LMS": {
        "from_config": LMSDiscreteScheduler.from_config
    },
    "PNDM": {
        "from_config": PNDMScheduler.from_config
    },
    "DDIM": {
        "from_config": DDIMScheduler.from_config
    },
    "K_EULER": {
        "from_config": EulerDiscreteScheduler.from_config
    },
    "K_EULER_ANCESTRAL": {
        "from_config": EulerAncestralDiscreteScheduler.from_config
    },
    "HEUN": {
        "from_config": HeunDiscreteScheduler.from_config
    },
    "DPM": {
        "from_config": DPMSolverMultistepScheduler.from_config
    },
    "DPM_SINGLESTEP": {
        "from_config": DPMSolverSinglestepScheduler.from_config
    }
}

SD_SCHEDULER_CHOICES = list(SD_SCHEDULERS.keys())
SD_SCHEDULER_DEFAULT = SD_SCHEDULER_CHOICES[0]
