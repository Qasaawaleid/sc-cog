from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE
import concurrent.futures
import boto3
import os

s3 = boto3.resource('s3',
                    endpoint_url=os.environ['ENDPOINT_URL'],
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                    )
bucket_name = 'models'


def download_sd_model(key):
    print(f"⏳ Downloading model: {key}")
    s3_directory = key + '/'
    # Specify the local directory to sync to
    local_directory = get_local_model_path(key) + '/'
    # Loop through all files in the S3 directory
    for object in s3.Bucket(bucket_name).objects.filter(Prefix=s3_directory):
        # Get the file key and local file path
        key = object.key
        local_file_path = os.path.join(
            local_directory, key.replace(s3_directory, ''))
        # Skip if the local file already exists and is the same size
        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == object.size:
            continue
        s3.download_file(bucket_name, key, local_file_path)
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


def get_local_model_path(key):
    return f"./{SD_MODEL_CACHE}/{SD_MODELS[key]['id']}"


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
