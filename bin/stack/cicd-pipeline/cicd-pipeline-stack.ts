import * as cdk from 'aws-cdk-lib';
import * as codepipeline from 'aws-cdk-lib/aws-codepipeline';
import * as codepipeline_actions from 'aws-cdk-lib/aws-codepipeline-actions';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

import { BaseStack, StackCommonProps } from '../../../lib/base/base-stack';

export class CicdPipelineStack extends BaseStack {
    private sourceOutput: codepipeline.Artifact;

    constructor(scope: Construct, props: StackCommonProps, stackConfig: any) {
        super(scope, stackConfig.Name, props, stackConfig);

        const {
            RepositoryName: repositoryName,
            BranchName: branchName,
            ConnectionArn: connectionArn,
        } = stackConfig;

        if (!repositoryName || !branchName || !connectionArn) {
            throw new Error("RepositoryName, BranchName, or ConnectionArn is missing in the stack configuration.");
        }

        // Define artifact bucket for pipeline
        const artifactBucket = s3.Bucket.fromBucketName(
            this,
            'ImportedArtifactBucket',
            cdk.Fn.importValue('BuildArtifactBucketName')
        );

        // Define the pipeline
        const pipeline = new codepipeline.Pipeline(this, 'CICDPipeline', {
            pipelineName: `${this.projectPrefix}-CICD-Pipeline`,
            artifactBucket: artifactBucket,
        });

        // Add Source Stage
        const sourceStage = pipeline.addStage({ stageName: 'Source' });
        sourceStage.addAction(this.createSourceStageAction(repositoryName, branchName, connectionArn));

        // Add Build Stage
        const buildStage = pipeline.addStage({ stageName: 'Build' });
        buildStage.addAction(this.createBuildStageAction('Build', 'script/cicd/buildspec_cdk_deploy.yml'));
    }

    private createSourceStageAction(repositoryName: string, branchName: string, connectionArn: string): codepipeline.IAction {
        // Define source artifact
        this.sourceOutput = new codepipeline.Artifact('SourceOutput');

        // Create CodeStar Connections Source Action
        return new codepipeline_actions.CodeStarConnectionsSourceAction({
            actionName: 'Get_Source',
            owner: 'Carma-tech', // Update with your GitHub organization or repository owner
            repo: repositoryName,
            branch: branchName,
            connectionArn: connectionArn,
            output: this.sourceOutput,
        });
    }

    private createBuildStageAction(actionName: string, buildSpecPath: string): codepipeline.IAction {
        // Validate buildspec file existence
        if (!buildSpecPath) {
            throw new Error(`BuildSpec file path is missing.`);
        }

        // Define CodeBuild project
        const project = new codebuild.PipelineProject(this, `${actionName}-Project`, {
            environment: {
                buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
                privileged: true,
                computeType: codebuild.ComputeType.MEDIUM,
            },
            environmentVariables: {
                PROJECT_PREFIX: { value: this.projectPrefix },
                REGION: { value: cdk.Stack.of(this).region },
                ACCOUNT_ID: { value: cdk.Stack.of(this).account },
            },
            buildSpec: codebuild.BuildSpec.fromSourceFilename(buildSpecPath),
        });

        // Attach IAM policies to the project role
        project.addToRolePolicy(this.getDeployCommonPolicy());
        project.addToRolePolicy(this.getServiceSpecificPolicy());

        // Define build artifact
        const buildOutput = new codepipeline.Artifact(`${actionName}BuildOutput`);

        // Create CodeBuild Action
        return new codepipeline_actions.CodeBuildAction({
            actionName,
            project,
            input: this.sourceOutput,
            outputs: [buildOutput],
        });
    }

    private getDeployCommonPolicy(): iam.PolicyStatement {
        return new iam.PolicyStatement({
            actions: [
                "cloudformation:*",
                "s3:*",
                "lambda:*",
                "ssm:*",
                "iam:*",
                "kms:*",
                "events:*",
            ],
            resources: ["*"],
        });
    }

    private getServiceSpecificPolicy(): iam.PolicyStatement {
        return new iam.PolicyStatement({
            actions: [
                "ec2:*",
                "cloudwatch:*",
                "sagemaker:*",
                "ses:*",
                "sns:*",
                "application-autoscaling:*",
                "apigateway:*",
                "logs:*",
                "elasticloadbalancingv2:*",
                "elasticloadbalancing:*",
            ],
            resources: ["*"],
        });
    }
}
