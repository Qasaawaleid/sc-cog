from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE


import concurrent.futures
import boto3
import os

s3 = boto3.resource('s3',
                    endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get(
                        'AWS_SECRET_ACCESS_KEY')
                    )
bucket_name = os.environ.get('S3_BUCKET_NAME')


def download_sd_model(model_id):
    print(f"⏳ Downloading model: {model_id}")
    s3_directory = model_id + '/'
    # Specify the local directory to sync to
    local_directory = get_local_model_path(model_id) + '/'
    bucket = s3.Bucket(bucket_name)
    # Loop through all files in the S3 directory
    for object in bucket.objects.filter(Prefix=s3_directory):
        # Get the file key and local file path
        key = object.key
        local_file_path = os.path.join(
            local_directory, key.replace(s3_directory, '')
        )
        # Skip if the local file already exists and is the same size
        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == object.size:
            continue
        # Create the local directory if it doesn't exist
        local_directory_path = os.path.dirname(local_file_path)
        if not os.path.exists(local_directory_path):
            os.makedirs(local_directory_path)
        print(f'Downloading to: {local_file_path}')
        bucket.download_file(key, local_file_path)
    print(f"✅ Downloaded model: {model_id}")
    return {
        "model_id": model_id
    }


def download_sd_models_concurrently():
    model_ids = []
    for key in SD_MODELS:
        model_ids.append(SD_MODELS[key]["id"])
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [executor.submit(
            download_sd_model, model_id) for model_id in model_ids]
        # Wait for all tasks to complete
        results = [task.result()
                   for task in concurrent.futures.as_completed(download_tasks)]


def get_local_model_path(model_id):
    return f"./{SD_MODEL_CACHE}/{model_id}"


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
