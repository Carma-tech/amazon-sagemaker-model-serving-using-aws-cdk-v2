import * as cdk from 'aws-cdk-lib';
import * as glue from 'aws-cdk-lib/aws-glue';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';
import * as path from 'path';
import { BucketDeployment, Source } from 'aws-cdk-lib/aws-s3-deployment';

export class GlueJobStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props?: cdk.StackProps) {
        super(scope, id, props);

        // S3 bucket for storing Glue scripts and model artifacts
        const bucket = s3.Bucket.fromBucketName(this, 'ModelBucket', 'textclassificationmldemo-model-archiving-us-east-1-1272');

        // Create IAM policy for the S3 bucket
        const bucketPolicy = new iam.PolicyStatement({
            actions: ['s3:GetObject', 's3:PutObject'],
            resources: [bucket.bucketArn + '/*'],
        });

        // Deploy scripts to Assets
        new BucketDeployment(this, 'DeployScript', {
            sources: [Source.asset(path.join(__dirname, '../../../notebook/glue_job'))],
            destinationBucket: bucket,
            destinationKeyPrefix: 'scripts'
        });

        // IAM role for Glue Job
        const glueRole = new iam.Role(this, 'GlueJobRole', {
            assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
            ],
        });

        // Glue Job
        const glueJob = new glue.CfnJob(this, 'ModelTrainingJob', {
            name: 'ModelTrainingJob',
            role: glueRole.roleArn,
            command: {
                name: 'glueetl',
                pythonVersion: '3',
                scriptLocation: `s3://${bucket.bucketName}/scripts/model_training.py`, // Path to your Python script
            },
            defaultArguments: {
                '--additional-python-modules': 'pyspark==3.1.1,boto3==1.20.32',
                '--job-language': 'python',
            },
            glueVersion: '4.0',
            workerType: 'G.1X',
            numberOfWorkers: 2,
            timeout: 120,
        });

        // Output the S3 bucket name
        new cdk.CfnOutput(this, 'GlueJobBucketName', {
            value: bucket.bucketName,
        });
    }
}
