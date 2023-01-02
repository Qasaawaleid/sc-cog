import torch
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler
)

SD_MODEL_CACHE = "diffusers-cache"

SD_MODELS = {
  "Stable Diffusion v1.5": {
    "id": "runwayml/stable-diffusion-v1-5",
    "revision": "fp16",
    "torch_dtype": torch.float16,
  },
  "Openjourney": {
    "id": "prompthero/openjourney",
    "prompt_prefix": "mdjrny-v4 style"
  },
  "Redshift Diffusion": {
    "id": "nitrosocke/redshift-diffusion",
    "prompt_prefix": "redshift style"
  },
  "Arcane Diffusion": {
    "id": "nitrosocke/Arcane-Diffusion",
    "prompt_prefix": "arcane style"
  },
  "Ghibli Diffusion": {
    "id": "nitrosocke/Ghibli-Diffusion",
    "prompt_prefix": "ghibli style"
  },
  "Mo-Di Diffusion": {
    "id": "nitrosocke/mo-di-diffusion",
    "prompt_prefix": "modern disney style"
  },
  "Waifu Diffusion v1.4": {
    "id": "hakurei/waifu-diffusion"
  },
  "22h Diffusion v0.1": {
    "id": "22h/vintedois-diffusion-v0-1",
    "prompt_prefix": "estilovintedois"
  },
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
  }
}

SD_SCHEDULERS_CHOICES = list(SD_SCHEDULERS.keys())
SD_SCHEDULER_DEFAULT = SD_SCHEDULERS_CHOICES[0]