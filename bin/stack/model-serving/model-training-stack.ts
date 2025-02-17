import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as cw_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as path from 'path';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as stepfunctions from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as cdk from 'aws-cdk-lib';
import { BaseStack, StackCommonProps } from '../../../lib/base/base-stack';
import { Construct } from 'constructs';
import { createCodePipeline } from '../utils/helper';


export class ModelTrainingStack extends BaseStack {

    constructor(scope: Construct, props: StackCommonProps, stackConfig: any) {
        super(scope, stackConfig.Name, props, stackConfig);

        // Get the bucket created in BaseStack
        const modelBucketName = this.getParameter('modelArchivingBucketName');
        const modelBucket = s3.Bucket.fromBucketName(this, 'ModelBucket', modelBucketName);

        // Create vpc
        const vpc: ec2.Vpc = new ec2.Vpc(this, 'VPC', {
            vpcName: 'model-training-vpc',
            maxAzs: 1,
            subnetConfiguration: [
                {
                    subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name: 'private',
                    cidrMask: 24,
                },
                {
                    subnetType: ec2.SubnetType.PUBLIC,
                    name: 'public',
                    cidrMask: 24,
                }
            ]
        });

        const privateSubnets = vpc.selectSubnets({
            subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        });

        // Create a new security group for batch jobs
        const batchSg = new ec2.SecurityGroup(this, 'BatchSecurityGroup', {
            vpc,
            description: 'Security group for Batch jobs',
            allowAllOutbound: true,
        });

        // Create an ECR repository
        const repository = new ecr.Repository(this, 'ModelTrainingRepository', {
            repositoryName: 'model-training-batch-repo',
            lifecycleRules: [{
                description: "Keeps a maximum number of images to minimize storage",
                maxImageCount: 5
            }],
            removalPolicy: cdk.RemovalPolicy.DESTROY
        });

        const pytorchRepo = ecr.Repository.fromRepositoryAttributes(this, 'PytorchTrainingRepo', {
            repositoryArn: `arn:aws:ecr:us-west-2:763104351884:repository/pytorch-training`,
            repositoryName: 'pytorch-training'
        });

        const pytorchImage = `${pytorchRepo.repositoryUri}:2.2.0-cpu-py310-ubuntu20.04-sagemaker-v1.37`;

        // Create cloudwatch log group for the Batch job
        const logGroup = new logs.LogGroup(this, 'BatchLogGroup', {
            logGroupName: '/aws/batch/model-training/job',
            retention: logs.RetentionDays.ONE_WEEK,
        });

        // Create an IAM Role for Batch Service
        const batchServiceRole = new iam.Role(this, 'BatchServiceRole', {
            assumedBy: new iam.ServicePrincipal('batch.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSBatchServiceRole'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
            ],
        });

        // Create an IAM Role for Batch Job
        const jobRole = new iam.Role(this, 'BatchJobRole', {
            assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSBatchServiceRole'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role')
            ],
        });

        jobRole.addToPolicy(new iam.PolicyStatement({
            actions: [
                'sagemaker:CreateTrainingJob',
                'sagemaker:DescribeTrainingJob',
                'sagemaker:StopTrainingJob',
                'sagemaker:ListTags',
                'sagemaker:AddTags',
                's3:GetObject',
                's3:PutObject',
                's3:ListBucket',
                'logs:*',

            ],
            resources: ['*'],
        }));

        // Create the EC2 instance role
        const ec2InstanceRole = new iam.Role(this, 'EC2InstanceRole', {
            assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore')
            ]
        });

        // Add additional required permissions for AWS Batch
        ec2InstanceRole.addToPolicy(new iam.PolicyStatement({
            effect: iam.Effect.ALLOW,
            actions: [
                'ecs:CreateCluster',
                'ecs:DeregisterContainerInstance',
                'ecs:DiscoverPollEndpoint',
                'ecs:Poll',
                'ecs:RegisterContainerInstance',
                'ecs:StartTelemetrySession',
                'ecs:Submit*',
                'ecr:GetAuthorizationToken',
                'ecr:BatchCheckLayerAvailability',
                'ecr:GetDownloadUrlForLayer',
                'ecr:BatchGetImage',
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
                'logs:DescribeLogStreams',
                's3:PutObject',
                's3:GetObject',
                's3:ListBucket',
            ],
            resources: ['*']
        }));

        // Create instance profile from the role
        const instanceProfile = new iam.CfnInstanceProfile(this, 'InstanceProfile', {
            roles: [ec2InstanceRole.roleName]
        });

        // IAM Role for SageMaker Training Job
        const sagemakerRole = new iam.Role(this, 'SageMakerTrainingRole', {
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
            ],
        });

        // Add additional permissions for accessing specific resources if needed
        sagemakerRole.addToPolicy(
            new iam.PolicyStatement({
                actions: [
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:StopTransformJob",
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:CreateHyperParameterTuningJob",
                    "sagemaker:DescribeHyperParameterTuningJob",
                    "sagemaker:StopHyperParameterTuningJob",
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:ListTags",
                    's3:GetObject',
                    's3:PutObject',
                    's3:ListBucket',
                    'logs:CreateLogStream',
                    'logs:PutLogEvents',
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer"
                ],
                resources: ['*'],
            })
        );

        jobRole.addToPolicy(new iam.PolicyStatement({
            actions: ['iam:PassRole'],
            resources: [sagemakerRole.roleArn]
        }))

        // Output the ARN of the created role for reference
        new cdk.CfnOutput(this, 'SageMakerRoleArn', {
            value: sagemakerRole.roleArn,
        });

        // Create a Job Definition
        const modelTrainingJobDefinition = new batch.CfnJobDefinition(this, 'ModelTrainingJobDefinition', {
            jobDefinitionName: 'ModelTrainingJobDefinition',
            type: 'container',
            containerProperties: {
                // image: '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker-v1.37',
                image: repository.repositoryUri,
                vcpus: 8,
                memory: 8192,
                jobRoleArn: jobRole.roleArn,
                command: ['python', 'models/model-a/src/code/sg_model.py'],
                logConfiguration: {
                    logDriver: 'awslogs',
                    options: {
                        'awslogs-group': logGroup.logGroupName,
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'model-training-batch'
                    }
                },
                environment: [
                    {
                        name: 'AWS_DEFAULT_REGION',
                        value: 'us-east-1'
                    },
                    {
                        name: 'S3_BUCKET',
                        value: modelBucket.bucketName
                    },
                    {
                        name: 'TRAIN_PREFIX',
                        value: 'models/model-a/training/data'
                    },
                    {
                        name: 'TEST_PREFIX',
                        value: 'models/model-a/training/test'
                    },
                    {
                        name: 'SG_ROLE',
                        value: sagemakerRole.roleArn
                    }

                ],

            },
        });

        // Create a Compute Environment
        const computeEnvironment = new batch.CfnComputeEnvironment(this, 'ModelTrainingComputeEnv', {
            computeEnvironmentName: 'ModelTrainingComputeEnv',
            type: 'MANAGED',
            state: 'ENABLED',
            computeResources: {
                type: 'EC2',
                minvCpus: 0,
                desiredvCpus: 8,
                maxvCpus: 32,
                instanceTypes: ['optimal'],
                securityGroupIds: [batchSg.securityGroupId],
                subnets: privateSubnets.subnetIds,
                instanceRole: instanceProfile.attrArn,
                allocationStrategy: 'BEST_FIT_PROGRESSIVE',
                ec2Configuration: [{
                    imageType: 'ECS_AL2',
                }]
            },
            // serviceRole: batchServiceRole.roleArn,
            serviceRole: `arn:aws:iam::${this.account}:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch`,
        });

        // Create a Job Queue
        const jobQueue = new batch.CfnJobQueue(this, 'ModelTrainingJobQueue', {
            jobQueueName: 'ModelTrainingJobQueue',
            priority: 1,
            state: 'ENABLED',
            computeEnvironmentOrder: [{
                order: 1,
                computeEnvironment: computeEnvironment.ref,
            }],
        });

        const batchTask = new tasks.BatchSubmitJob(this, 'SubmitBatchJob', {
            jobDefinitionArn: modelTrainingJobDefinition.ref,
            jobQueueArn: jobQueue.ref,
            jobName: 'ModelTrainingJob',
            resultPath: '$.batchResult',

        });

        // Create a state machine for the batch job
        const definition = stepfunctions.Chain
            .start(batchTask)
            .next(new stepfunctions.Choice(this, 'Job Complete?')
                .when(stepfunctions.Condition.stringEquals('$.batchResult.Status', 'SUCCEEDED'),
                    new stepfunctions.Succeed(this, 'Job Succeeded'))
                .otherwise(new stepfunctions.Fail(this, 'Job Failed')));

        const stateMachine = new stepfunctions.StateMachine(this, 'ModelTrainingBatchJobStateMachine', {
            definitionBody: stepfunctions.DefinitionBody.fromChainable(definition),
            timeout: cdk.Duration.hours(5),
        });

        new cdk.CfnOutput(this, 'BatchJobStateMachineArn', {
            value: stateMachine.stateMachineArn,
        });

        //////////////////////////////////////////////////////////////////
        // Define the SageMaker training job taks
        const trainingJobTask = new tasks.SageMakerCreateTrainingJob(this, 'TrainModel', {
            trainingJobName: 'ModelTrainingJob',
            role: sagemakerRole,
            algorithmSpecification: {
                trainingImage: tasks.DockerImage.fromRegistry('811284229777.dkr.ecr.us-east-1.amazonaws.com/blazingtext:1'),
                // algorithmName: 'blazingtext',
                trainingInputMode: tasks.InputMode.FILE,
            },
            inputDataConfig: [
                {
                    channelName: 'train',
                    dataSource: {
                        s3DataSource: {
                            s3Location: tasks.S3Location.fromBucket(modelBucket, 'training-data/train'),
                            s3DataType: tasks.S3DataType.S3_PREFIX,
                        }
                    }
                },
                {
                    channelName: 'validation',
                    dataSource: {
                        s3DataSource: {
                            s3Location: tasks.S3Location.fromBucket(modelBucket, 'training-data/validation'),
                            s3DataType: tasks.S3DataType.S3_PREFIX,
                        }
                    }
                }
            ],
            outputDataConfig: {
                s3OutputLocation: tasks.S3Location.fromBucket(modelBucket, 'train/output'),
            },
            resourceConfig: {
                instanceCount: 1,
                instanceType: new ec2.InstanceType('m5.2xlarge'),
                volumeSize: cdk.Size.gibibytes(10),
            },
            stoppingCondition: {
                maxRuntime: cdk.Duration.hours(1),
            },
        });

        // Define step function workflow
        const trainingJobDefinition = stepfunctions.Chain.start(trainingJobTask)
            .next(
                new stepfunctions.Choice(this, 'Is Training Successful?')
                    .when(
                        stepfunctions.Condition.stringEquals('$.TrainingJobStatus', 'Failed'),
                        new stepfunctions.Fail(this, 'Training Failed')
                    )
                    .otherwise(new stepfunctions.Succeed(this, 'Training Succeeded'))
            );

        const trainingTaskStatemachine = new stepfunctions.StateMachine(this, 'SageMakerTrainingJobStateMachine', {
            definitionBody: stepfunctions.DefinitionBody.fromChainable(trainingJobDefinition),
            timeout: cdk.Duration.hours(1)
        })

        // Create codepipeline 
        createCodePipeline({
            scope: this,
            pipelineName: 'SageMakerModelServingPipeline',
            owner: 'Carma-tech',
            sourceRepo: 'amazon-sagemaker-model-serving-using-aws-cdk-v2',
            sourceBranch: 'add-model-training-stack',
            connectionArn: 'arn:aws:codeconnections:us-east-1:717918134056:connection/7e4bcd1d-6aea-4dee-98e5-edf22f6cadb0',
            ecrRepository: repository,
            dockerFolder: './',
            buildSpecPath: path.join(__dirname, '../../../buildspec.yml'),
            containerName: 'SageMakerModelServingContainer',
            jobDefinition: modelTrainingJobDefinition
        })

    }

}
