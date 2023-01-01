import torch
SD_MODEL_CACHE = "diffusers-cache"

SD_MODELS = {
  "Stable Diffusion v1.5": {
    "id": "runwayml/stable-diffusion-v1-5",
    "revision": "fp16",
    "torch_dtype": torch.float16
  },
  "Openjourney": {
    "id": "prompthero/openjourney"
  },
  "Redshift Diffusion": {
    "id": "nitrosocke/redshift-diffusion"
  },
  "Arcane Diffusion": {
    "id": "nitrosocke/Arcane-Diffusion"
  },
  "Ghibli Diffusion": {
    "id": "nitrosocke/Ghibli-Diffusion"
  },
  "Mo-Di Diffusion": {
    "id": "nitrosocke/mo-di-diffusion"
  },
  "Waifu Diffusion v1.4": {
    "id": "hakurei/waifu-diffusion"
  },
}

SD_MODEL_CHOICES = list(SD_MODELS.keys())
SD_MODEL_DEFAULT = SD_MODEL_CHOICES[0]