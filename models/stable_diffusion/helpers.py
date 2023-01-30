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


def download_sd_model(key):
    model_id = SD_MODELS[key]['id']
    print(f"⏳ Downloading model: {model_id}")
    s3_directory = model_id + '/'
    # Specify the local directory to sync to
    local_directory = get_local_model_path(key) + '/'
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
        print(f'Downloading "{key}" to "{local_file_path}"...')
        bucket.download_file(key, local_file_path)
    print(f"✅ Downloaded model: {key}")
    return {
        "key": key
    }


def download_sd_models_concurrently():
    with concurrent.futures.ThreadPoolExecutor(6) as executor:
        # Start the download tasks
        download_tasks = [executor.submit(
            download_sd_model, key) for key in SD_MODELS]
        # Wait for all tasks to complete
        results = [task.result()
                   for task in concurrent.futures.as_completed(download_tasks)]


def get_local_model_path(key):
    return f"./{SD_MODEL_CACHE}/{SD_MODELS[key]['id']}"


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
