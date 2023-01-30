import os
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError
from stable_diffusion.constants import SD_MODELS_ALL
import shutil

SD_MODEL_IDS = [model["id"] for model in SD_MODELS_ALL.values()]
REPO = "https://huggingface.co"
REPO_LIST = []
for model_id in SD_MODEL_IDS:
    REPO_LIST.append(REPO + "/" + model_id)
PREFIX = ""
FILE_LIST = [
    "feature_extractor/", "safety_checker/",
    "scheduler", "text_encoder", "tokenizer", "unet", "vae", "model_index.json"
]


def clone_repo(repo_url, repo_name):
    subprocess.call(["git", "clone", repo_url, repo_name])


def clear_bucket(s3):
    bucket = os.environ['S3_BUCKET_NAME']
    s3.delete_objects(
        Bucket=bucket,
        Delete={'Objects': [{'Key': k} for k in [
            obj['Key'] for obj in s3.list_objects(Bucket=bucket)['Contents']]]
        }
    )


def upload_to_s3(s3, repo_name, prefix, file):
    file_path = repo_name + "/" + file
    if os.path.isdir(file_path):
        for root, dirs, files in os.walk(file_path):
            for filename in files:
                local_path = os.path.join(root, filename)
                s3_path = prefix + repo_name + "/" + \
                    local_path[len(repo_name)+1:]
                print(f"Uploading to: {s3_path}")
                s3.upload_file(
                    local_path, os.environ['S3_BUCKET_NAME'], s3_path)
    else:
        s3_path = prefix + repo_name + "/" + file
        print(f"Uploading to: {s3_path}")
        s3.upload_file(
            file_path, os.environ['S3_BUCKET_NAME'], s3_path)


def main(repo_list, prefix="", file_list=None):
    try:
        s3 = boto3.client('s3',
                          endpoint_url=os.environ['ENDPOINT_URL'],
                          aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                          aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                          )
        clear_bucket(s3)
        for repo in repo_list:
            repo_name = repo.split("/")[-1].split(".")[0]
            clone_repo(repo, repo_name)
            if file_list:
                for file in file_list:
                    upload_to_s3(s3, repo_name, prefix, file)
            else:
                upload_to_s3(s3, repo_name, prefix, repo_name)
            shutil.rmtree(repo_name)
    except NoCredentialsError as e:
        print("Error:", e)


if __name__ == "__main__":
    main(REPO_LIST, PREFIX, FILE_LIST)
