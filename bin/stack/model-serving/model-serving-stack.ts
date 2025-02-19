import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import { BaseStack, StackCommonProps } from '../../../lib/base/base-stack';
import { Construct } from 'constructs';

interface ModelProps {
    modelName: string;
    role: iam.IRole;
    modelBucketName: string;
    modelS3Key: string;
    modelDockerImage: string;
    modelServerWorkers: string;
}

interface VariantConfigProps {
    variantName: string;
    variantWeight: number;
    modelName: string;
    serverlessConfig: sagemaker.CfnEndpointConfig.ServerlessConfigProperty; // Use ServerlessConfigProperty
}

interface EndpointConfigProps {
    endpointConfigName: string;
    role: iam.IRole;
    variantConfigPropsList: VariantConfigProps[];
}

interface EndpointProps {
    endpointName: string;
    endpointConfigName: string;
}

export class ModelServingStack extends BaseStack {
    constructor(scope: Construct, props: StackCommonProps, stackConfig: any) {
        super(scope, stackConfig.Name, props, stackConfig);

        const role: iam.IRole = this.createIamRole(`ModelEndpoint-Role`);

        const modelBucketName: string = this.getParameter('modelArchivingBucketName');
        let modelConfigList: VariantConfigProps[] = [];
        const modelList: any[] = stackConfig.ModelList;

        for (let model of modelList) {
            const modelName = this.createModel({
                modelName: model.ModelName,
                modelDockerImage: model.ModelDockerImage,
                modelS3Key: model.ModelS3Key,
                modelBucketName: modelBucketName,
                role: role,
                modelServerWorkers: model.ModelServerWorkers
            });

            // Define the serverless configuration
            const serverlessConfig: sagemaker.CfnEndpointConfig.ServerlessConfigProperty = {
                maxConcurrency: model.ServerlessConfig.MaxConcurrency, // From config
                memorySizeInMb: model.ServerlessConfig.MemorySizeInMb // From config
            };

            modelConfigList.push({
                modelName: modelName,
                variantName: model.VariantName,
                variantWeight: model.VariantWeight,
                serverlessConfig: serverlessConfig // Pass the serverless config
            });
        }

        const endpointConfigName = this.createEndpointConfig({
            endpointConfigName: stackConfig.EndpointConfigName,
            variantConfigPropsList: modelConfigList,
            role: role
        });

        let endpointName = ' ';
        if (stackConfig.Deploy) {
            endpointName = this.deployEndpoint({
                endpointName: stackConfig.EndpointName,
                endpointConfigName: endpointConfigName
            });
        }

        this.putParameter('sageMakerEndpointName', endpointName);
    }

    private createModel(props: ModelProps): string {
        const model = new sagemaker.CfnModel(this, `${props.modelName}-Model`, {
            modelName: `${this.projectPrefix}-${props.modelName}-Model`,
            executionRoleArn: props.role.roleArn,
            containers: [
                {
                    image: props.modelDockerImage,
                    modelDataUrl: `s3://${props.modelBucketName}/${props.modelS3Key}/model.tar.gz`,
                    environment: {
                        SAGEMAKER_MODEL_SERVER_WORKERS: props.modelServerWorkers,
                        SAGEMAKER_MODEL_SERVER_TIMEOUT: "3600",
                        SAGEMAKER_DEFAULT_INVOCATIONS_TIMEOUT: "3600"
                    }
                }
            ]
        });
        return model.attrModelName;
    }

    private createEndpointConfig(props: EndpointConfigProps): string {
        const endpointConfig = new sagemaker.CfnEndpointConfig(this, `${props.endpointConfigName}-Config`, {
            endpointConfigName: `${this.projectPrefix}-${props.endpointConfigName}-Config`,
            productionVariants: props.variantConfigPropsList.map(modelConfig => {
                return {
                    modelName: modelConfig.modelName,
                    variantName: modelConfig.variantName,
                    initialVariantWeight: modelConfig.variantWeight,
                    serverlessConfig: modelConfig.serverlessConfig // Use the serverless config
                };
            })
            // Remove dataCaptureConfig for serverless endpoints
        });

        return endpointConfig.attrEndpointConfigName;
    }

    private deployEndpoint(props: EndpointProps): string {
        const endpointName = `${this.projectPrefix}-${props.endpointName}-Endpoint`;
        const endpoint = new sagemaker.CfnEndpoint(this, `${props.endpointName}-Endpoint`, {
            endpointName: endpointName,
            endpointConfigName: props.endpointConfigName
        });

        return endpointName;
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
                                "cloudwatch:PutMetricData",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                                "logs:CreateLogGroup",
                                "logs:DescribeLogStreams",
                                "ec2:CreateNetworkInterface",
                                "ec2:CreateNetworkInterfacePermission",
                                "ec2:DeleteNetworkInterface",
                                "ec2:DeleteNetworkInterfacePermission",
                                "ec2:DescribeNetworkInterfaces",
                                "ec2:DescribeVpcs",
                                "ec2:DescribeDhcpOptions",
                                "ec2:DescribeSubnets",
                                "ec2:DescribeSecurityGroups"
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
}