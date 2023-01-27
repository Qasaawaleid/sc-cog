import os


def download_swinir_model(model_url):
    print(f"⏳ Downloading SwinIR model: {model_url}...")
    os.system(f'wget {model_url} -P experiments/pretrained_models')
    print(f"✅ Downloaded SwinIR model: {model_url}")
