from sagemaker.pytorch import PyTorch
import sagemaker
import boto3
import yaml
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# role = 'arn:aws:iam::717918134056:role/ModelTrainingStack-SageMakerTrainingRoleD9E95376-ZXyeTOixQVzi'

# Initialize S3 client
s3 = boto3.client('s3')

# Define S3 bucket and prefixes
bucket_name = os.getenv(
    'BUCKET_NAME', 'textclassificationmldemo-model-archiving')
train_prefix = os.getenv('TRAIN_PREFIX', 'models/model-a/training/data')
test_prefix = os.getenv('TEST_PREFIX', 'models/model-a/training/test')
role = os.getenv(
    'SG_ROLE', 'arn:aws:iam::717918134056:role/TextClassificationMLDemo--SageMakerTrainingRoleD9E9-QSutTBF1WfF0')

logger.info(
    f'Bucket name: {bucket_name}, train_prefix: {train_prefix}, test_prefix: {test_prefix}')
logger.info(f'Role: {role}')

# Define a function to create folders (prefixes) in the S3 bucket


def create_s3_folders(bucket_name, prefixes):
    """
    Creates folders (prefixes) in the specified S3 bucket.

    :param bucket_name: Name of the S3 bucket
    :param prefixes: List of prefixes (folder paths) to create in the bucket
    """
    for prefix in prefixes:
        if not prefix.endswith('/'):
            prefix += '/'
        # Create an empty object with a trailing slash to simulate a folder
        s3.put_object(Bucket=bucket_name, Key=prefix)
        logger.info(f"Created folder: s3://{bucket_name}/{prefix}")


# Create train and test folders in the S3 bucket
create_s3_folders(bucket_name, [train_prefix, test_prefix])

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the YAML configuration file
config_path = os.path.join(
    script_dir, '../../../../config/model-traning-config.yaml')

logger.info(f'config path: {config_path}')

if not os.path.exists(config_path):
    logger.info(f"Configuration file does not exist at: {config_path}")
else:
    logger.info(f"Configuration file found at: {config_path}")

# Verify if the file exists
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

# Load YAML configuration
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract training parameters
training_params = config["Training"]["Parameters"]
resources = config["Training"]["Resources"]
output = config["Training"]["Output"]
logger.info(
    f'training params: {training_params} \nresource: {resources} \noutput: {output}')

# Dynamically set PyTorch Estimator parameters from YAML
pytorch_estimator = PyTorch(
    entry_point="train.py",  # Training script
    # Directory containing train.py and dependencies (e.g., model.py)
    source_dir=script_dir,
    dependencies=[os.path.join(script_dir, "requirements.txt")],
    role=role,
    instance_type=resources["instance_type"],
    instance_count=resources["instance_count"],
    framework_version="2.3.0",
    py_version="py311",
    script_mode=True,
    hyperparameters={
        "epochs": training_params["num_epochs"],
        "batch-size": training_params["batch_size"],
        "learning-rate": training_params["learning_rate"],
    },
    use_spot_instances=True,
    max_wait=training_params["max_wait"],
    max_run=training_params['max_run'],
)

pytorch_estimator.fit({'train': f's3://{bucket_name}/{train_prefix}',
                       'test': f's3://{bucket_name}/{test_prefix}'
                       })
logger.info("Training completed.")
