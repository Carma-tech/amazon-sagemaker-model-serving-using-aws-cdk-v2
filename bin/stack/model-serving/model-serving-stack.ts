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
  // For real-time (server-based) async inference, the instanceCount and instanceType are required.
  instanceCount?: number;
  instanceType?: string;
  // For real-time (serverless) inference, you might use a serverlessConfig.
  // When using async inference, omit this property.
  serverlessConfig?: sagemaker.CfnEndpointConfig.ServerlessConfigProperty;
}

interface EndpointConfigProps {
  endpointConfigName: string;
  role: iam.IRole;
  variantConfigPropsList: VariantConfigProps[];
  // Optional asynchronous inference configuration.
  asyncInferenceConfig?: sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty;
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

    // Create each model and add its variant configuration.
    for (let model of modelList) {
      const createdModelName = this.createModel({
        modelName: model.ModelName,
        modelDockerImage: model.ModelDockerImage,
        modelS3Key: model.ModelS3Key,
        modelBucketName: modelBucketName,
        role: role,
        modelServerWorkers: model.ModelServerWorkers,
      });

      // Define the serverless configuration for the model. This is required for serverless inference. comment out when using async inference.
      //   const serverlessConfig: sagemaker.CfnEndpointConfig.ServerlessConfigProperty = {
      //     maxConcurrency: model.ServerlessConfig.MaxConcurrency, // From config
      //     memorySizeInMb: model.ServerlessConfig.MemorySizeInMb // From config
      // };

      modelConfigList.push({
        modelName: createdModelName,
        variantName: model.VariantName,
        variantWeight: model.VariantWeight,
        instanceCount: model.InstanceCount,
        instanceType: model.InstanceType,

        // Omit serverlessConfig when async inference is used.
        // serverlessConfig: serverlessConfig // Pass the serverless config
      });
    }

    // Define the async inference configuration.
    // This config tells SageMaker to queue requests and write output to the specified S3 path.
    const asyncConfig: sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty = {
      outputConfig: {
        s3OutputPath: `s3://${modelBucketName}/model/async-output/`,
        // Optionally, you can add NotificationConfig here.
      },
      // Optionally, you can include ClientConfig:
      // clientConfig: { MaxConcurrentInvocationsPerInstance: 4 },
    };

    // Create the endpoint configuration using the async config.
    // Ensure that you use a consistent endpoint config name.
    const endpointConfigName = this.createEndpointConfig({
      endpointConfigName: stackConfig.EndpointConfigName,
      variantConfigPropsList: modelConfigList,
      role: role,
      asyncInferenceConfig: asyncConfig,
    });

    let endpointName = ' ';
    if (stackConfig.Deploy) {
      // Deploy the endpoint using the same endpoint configuration name.
      endpointName = this.deployEndpoint({
        endpointName: stackConfig.EndpointName,
        endpointConfigName: endpointConfigName,
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
            SAGEMAKER_DEFAULT_INVOCATIONS_TIMEOUT: "3600",
          },
        },
      ],
    });
    return model.attrModelName;
  }

  private createEndpointConfig(props: EndpointConfigProps): string {
    // Build production variants. When async inference is used,
    // we omit any serverlessConfig property.
    const productionVariants = props.variantConfigPropsList.map(modelConfig => ({
      modelName: modelConfig.modelName,
      variantName: modelConfig.variantName,
      initialVariantWeight: modelConfig.variantWeight,
      // For real-time (server-based) async inference, the instanceCount and instanceType are required.
      // When using serverless inference then comment these two options
      instanceType: modelConfig.instanceType,
      initialInstanceCount: modelConfig.instanceCount,
      // serverlessConfig: modelConfig.serverlessConfig, // Omit this property when using async inference.
    }));

    const endpointConfig = new sagemaker.CfnEndpointConfig(this, `${props.endpointConfigName}-Config`, {
      endpointConfigName: `${this.projectPrefix}-${props.endpointConfigName}-Config`,
      productionVariants: productionVariants,
      // Comment out this when using serverless inference.
      asyncInferenceConfig: props.asyncInferenceConfig, // Enables asynchronous inference.
    });

    return endpointConfig.attrEndpointConfigName;
  }

  private deployEndpoint(props: EndpointProps): string {
    const endpointName = `${this.projectPrefix}-${props.endpointName}-Endpoint`;
    new sagemaker.CfnEndpoint(this, `${props.endpointName}-Endpoint`, {
      endpointName: endpointName,
      endpointConfigName: props.endpointConfigName,
    });
    return endpointName;
  }

  private createIamRole(roleBaseName: string): iam.IRole {
    const role = new iam.Role(this, roleBaseName, {
      roleName: `${this.projectPrefix}-${roleBaseName}`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        { managedPolicyArn: 'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess' },
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
              resources: ['*'],
            }),
          ],
        }),
      },
    });

    role.addManagedPolicy({ managedPolicyArn: 'arn:aws:iam::aws:policy/AmazonS3FullAccess' });

    return role;
  }
}
