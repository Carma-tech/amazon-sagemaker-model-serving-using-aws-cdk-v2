import boto3
import json
import time
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run SageMaker Batch Transform Job')
    parser.add_argument(
        '--input-text',
        type=str,
        required=True,
        help='Input text for classification (e.g. "The new president has called for an emergency conference.")'
    )

    return parser.parse_args()


def upload_test_data(bucket_name: str, input_text: str) -> str:
    s3 = boto3.client('s3')

    # Test data
    test_data = {
        "sentence": input_text
    }

    # Generate unique filename
    timestamp = int(time.time())
    file_key = f'models/model-a/input/test_{timestamp}.json'

    try:
        # Upload to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=json.dumps(test_data),
            ContentType='application/json'
        )
        print(
            f"Successfully uploaded test data to s3://{bucket_name}/{file_key}")
        return file_key
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        sys.exit(1)


def start_transform_job(state_machine_arn: str) -> str:
    sfn = boto3.client('stepfunctions')

    # Generate a unique job name using timestamp
    timestamp = int(time.time())
    unique_job_name = f"transform-job-{timestamp}"

    # Start the state machine execution
    response = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=f'transform-{timestamp}',
        input=json.dumps({
            "TransformJobName": unique_job_name
        })
    )
    return response['executionArn']


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Get parameters from SSM
    ssm = boto3.client('ssm')

    # input_bucket = ssm.get_parameter(
    #     Name='textclassification-model-transform'
    # )['Parameter']['Value']
    input_bucket = 'textclassificationmldemo-model-transform-us-east-1-1272'

    # state_machine_arn = ssm.get_parameter(
    #     Name='/your-project/transformStateMachineArn-YourModelName'
    # )['Parameter']['Value']
    state_machine_arn = 'arn:aws:states:us-east-1:127214197188:stateMachine:TextClassificationMLDemo-Model-A-20241218a-Transform-StateMachine'

    # Upload test data
    file_key = upload_test_data(input_bucket, args.input_text)

    # Start transform job
    timestamp = int(time.time())
    job_name = f'text-classification-{timestamp}'

    execution_arn = start_transform_job(
        state_machine_arn
    )
    
    print("\nBatch Transform Job Started!")
    print(f"Input file: s3://{input_bucket}/{job_name}")
    print(f"Job name: {job_name}")
    print(
        f"Check the AWS Step Functions console for job status using execution ARN: {execution_arn}")


if __name__ == '__main__':
    main()
