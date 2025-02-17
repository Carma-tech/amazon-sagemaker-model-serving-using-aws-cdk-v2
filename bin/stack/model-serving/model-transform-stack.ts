import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { BaseStack, StackCommonProps } from '../../../lib/base/base-stack';
import { v4 as uuidv4 } from 'uuid';


interface TransformJobProps {
    jobName: string;
    modelName: string;
    instanceType: ec2.InstanceType;
    inputBucketName: string;
    outputBucketName: string;
}


export class ModelTransformJobStack extends BaseStack {
    constructor(scope: Construct, props: StackCommonProps, stackConfig: any) {
        super(scope, stackConfig.Name, props, stackConfig);

        const role: iam.IRole = this.createIamRole(`TransformJob-Role`);

        const transformBucket = this.createS3Bucket(`${stackConfig.BucketBaseName}`);

        // Get the model list from stack config
        const modelList: any[] = stackConfig.ModelList;
        const timestamp = new Date().getTime();

        // Create transform jobs for each model
        for (let model of modelList) {
            const stateMachine = this.createBatchTransformJob({
                jobName: `${this.projectPrefix}-${model.ModelName}-Transform`,
                modelName: `${this.projectPrefix}-${model.ModelName}-Model`,
                instanceType: model.InstanceType,
                inputBucketName: `${transformBucket.bucketName}`,
                outputBucketName: `${transformBucket.bucketName}`,
            });

            this.putParameter(
                `transformStateMachineArn-${model.ModelName}`,
                stateMachine.stateMachineArn
            );
        }

        // Grand S3 permissions
        transformBucket.grantReadWrite(role);

        // Store bucket name in parameters
        this.putParameter('textclassification-model-transform', transformBucket.bucketName)
    }

    private createIamRole(roleBaseName: string): iam.IRole {
        const role = new iam.Role(this, roleBaseName, {
            roleName: `${this.projectPrefix}-${roleBaseName}`,
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            managedPolicies: [
                { managedPolicyArn: 'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess' }
            ],
            inlinePolicies: {
                CloudWatchLogsAccess: new iam.PolicyDocument({
                    statements: [
                        new iam.PolicyStatement({
                            effect: iam.Effect.ALLOW,
                            actions: [
                                'cloudwatch:PutMetricData',
                                'logs:CreateLogStream',
                                'logs:PutLogEvents',
                                'logs:CreateLogGroup',
                                'logs:DescribeLogStreams',
                                "ec2:CreateNetworkInterface",
                                "ec2:CreateNetworkInterfacePermission",
                                "ec2:DeleteNetworkInterface",
                                "ec2:DeleteNetworkInterfacePermission",
                                "ec2:DescribeNetworkInterfaces",
                                "ec2:DescribeVpcs",
                                "ec2:DescribeDhcpOptions",
                                "ec2:DescribeSubnets",
                                "ec2:DescribeSecurityGroups",

                                'sagemaker:CreateTransformJob',
                                'sagemaker:DescribeTransformJob',
                                'sagemaker:StopTransformJob',
                                'sagemaker:ListTags',
                                'ecr:GetAuthorizationToken',
                                'ecr:BatchCheckLayerAvailability',
                                'ecr:GetDownloadUrlForLayer',
                                'ecr:BatchGetImage',
                                's3:GetObject',
                                's3:PutObject',
                                's3:ListBucket',
                            ],
                            resources: ['*']
                        })
                    ]
                })
            }
        });

        role.addManagedPolicy({ managedPolicyArn: 'arn:aws:iam::aws:policy/AmazonS3FullAccess' });

        return role;
    }

    private createBatchTransformJob(props: TransformJobProps): sfn.StateMachine {
        // Create the transform job task
        const transformJob = new tasks.SageMakerCreateTransformJob(this, `${props.jobName}`, {
            transformJobName: sfn.JsonPath.format(
                'transform-{}',
                sfn.JsonPath.stringAt('$$.Execution.Id').split(':').slice(-1)[0].replace(/[^a-zA-Z0-9]/g, '').substring(0, 30)
            ),
            modelName: props.modelName,
            modelClientOptions: {
                invocationsMaxRetries: 3,
                invocationsTimeout: cdk.Duration.minutes(5),
            },
            transformInput: {
                transformDataSource: {
                    s3DataSource: {
                        s3Uri: `s3://${props.inputBucketName}/models/model-a/input`,
                        s3DataType: tasks.S3DataType.S3_PREFIX,
                    }
                },
                contentType: 'application/json',
                splitType: tasks.SplitType.LINE,
            },
            transformOutput: {
                s3OutputPath: `s3://${props.outputBucketName}/models/model-a/output`,
                accept: 'application/json'
            },
            transformResources: {
                instanceCount: 1,
                // instanceType: ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.XLARGE)
                instanceType: props.instanceType || ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE)
            },
        });

        // Create state machine
        return new sfn.StateMachine(this, `${props.jobName}-StateMachine`, {
            stateMachineName: `${props.jobName}-StateMachine`,
            definitionBody: sfn.DefinitionBody.fromChainable(transformJob),
        });
    }

}
