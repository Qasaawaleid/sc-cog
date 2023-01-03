import os
import torch

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
            "torch_dtype": torch.float16,
        }
    elif model_env == "Openjourney":
        SD_MODELS[model_env] = {
            "id": "prompthero/openjourney",
            "prompt_prefix": "mdjrny-v4 style"
        }
    elif model_env == "Redshift Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/redshift-diffusion",
            "prompt_prefix": "redshift style"
        }
    elif model_env == "Arcane Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/Arcane-Diffusion",
            "prompt_prefix": "arcane style"
        }
    elif model_env == "Ghibli Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/Ghibli-Diffusion",
            "prompt_prefix": "ghibli style"
        }
    elif model_env == "Mo-Di Diffusion":
        SD_MODELS[model_env] = {
            "id": "nitrosocke/mo-di-diffusion",
            "prompt_prefix": "modern disney style"
        }
    elif model_env == "Waifu Diffusion v1.4":
        SD_MODELS[model_env] = {
            "id": "hakurei/waifu-diffusion",
            "prompt_prefix": "masterpiece, best quality, high quality",
            "negative_prompt_prefix": "worst quality, low quality, deleted, nsfw"
        }
    elif model_env == "22h Diffusion v0.1":
        SD_MODELS[model_env] = {
            "id": "22h/vintedois-diffusion-v0-1",
            "prompt_prefix": "estilovintedois"
        }

SD_MODEL_CHOICES = list(SD_MODELS.keys())
SD_MODEL_DEFAULT = SD_MODEL_CHOICES[0]

print(SD_MODELS)
print(SD_MODEL_CHOICES)
print(SD_MODEL_DEFAULT)
