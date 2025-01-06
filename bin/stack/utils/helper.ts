import { Pipeline, Artifact } from 'aws-cdk-lib/aws-codepipeline';
import { CodeBuildAction, CodeStarConnectionsSourceAction, EcsDeployAction, LambdaInvokeAction } from 'aws-cdk-lib/aws-codepipeline-actions';
import { Construct } from 'constructs';
import * as cdk from 'aws-cdk-lib';
import { ManagedPolicy, PolicyStatement } from 'aws-cdk-lib/aws-iam';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import { Repository, IRepository } from 'aws-cdk-lib/aws-ecr';
import { IFunction } from 'aws-cdk-lib/aws-lambda';
import { FargateService } from 'aws-cdk-lib/aws-ecs';
import { ArtifactPath } from 'aws-cdk-lib/aws-codepipeline';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';
import * as crypto from 'crypto';
import { CfnJobDefinition } from 'aws-cdk-lib/aws-batch';


interface CodePipelineProps {
    scope: Construct;
    pipelineName: string;
    owner: string;
    sourceRepo: string;
    sourceBranch: string;
    connectionArn: string;
    ecrRepository: IRepository;
    dockerFolder: string;
    buildSpecPath: string;
    containerName: string;
    ecsService?: FargateService;
    jobDefinition?: CfnJobDefinition;
}

interface SourceStageProps {
    pipeline: Pipeline;
    owner: string;
    repo: string;
    branch: string;
    connectionArn: string;
}

type BuildProjectEnvVars = {
    REPOSITORY_URI: { value: string };
    AWS_ACCOUNT_ID: { value: string };
    AWS_DEFAULT_REGION: { value: string };
    CONTAINER_NAME?: { value: string }; // Optional property for future use
};

interface CodePipelineProps {
    scope: Construct;
    pipelineName: string;
    owner: string;
    sourceRepo: string;
    sourceBranch: string;
    connectionArn: string;
    ecrRepository: IRepository;
    dockerFolder: string;
    buildSpecPath: string;
    containerName: string;
    lambdaFunction?: IFunction;
    ecsService?: FargateService;
    dbAccessPolicy?: PolicyStatement;
}

export function createCodePipeline(props: CodePipelineProps): void {
    const {
        scope,
        pipelineName,
        owner,
        sourceRepo,
        sourceBranch,
        connectionArn,
        ecrRepository,
        dockerFolder,
        buildSpecPath,
        containerName,
        ecsService,
        jobDefinition,
    } = props;

    // Import the artifact S3 bucket from the common-infra-stack
    const artifactBucket = s3.Bucket.fromBucketName(scope, 'ImportedArtifactBucket',
        cdk.Fn.importValue('BuildArtifactBucketName')
    );

    const pipeline = new Pipeline(scope, `${pipelineName}Pipeline`, {
        pipelineName: pipelineName,
        artifactBucket: artifactBucket
    });

    // Source Stage
    const sourceArtifact = createSourceStage({
        pipeline,
        owner,
        repo: sourceRepo,
        branch: sourceBranch,
        connectionArn,
    });

    const buildProject = createCodeBuildProject(scope, props);

    // Build Stage
    const buildOutput = new Artifact();
    const buildAction = new CodeBuildAction({
        actionName: "Build",
        input: sourceArtifact,
        project: buildProject,
        outputs: [buildOutput],
    });

    pipeline.addStage({
        stageName: "Build",
        actions: [buildAction],
    });

    if (ecsService) {
        const deployAction = new EcsDeployAction({
            actionName: 'ECS_Deploy',
            service: ecsService,
            imageFile: new ArtifactPath(buildOutput, 'imagedefinitions.json'),
        });

        pipeline.addStage({
            stageName: "Deploy",
            actions: [deployAction],
        });
    }

    pipeline.role.addToPrincipalPolicy(new PolicyStatement({
        actions: [
            'codestar-connections:UseConnection',
            'codestar-connections:GetConnection',
            'codestar-connections:ListConnections',
        ],
        resources: [connectionArn],
    }));

}


function createSourceStage(props: SourceStageProps): Artifact {
    const sourceArtifact = new Artifact();
    const sourceAction = new CodeStarConnectionsSourceAction({
        actionName: 'Get_Source',
        owner: props.owner,
        repo: props.repo,
        connectionArn: props.connectionArn,
        output: sourceArtifact,
        branch: props.branch,
    });

    props.pipeline.addStage({
        stageName: "Source",
        actions: [sourceAction],
    });

    return sourceArtifact;
}


function createCodeBuildProject(scope: Construct, props: CodePipelineProps): codebuild.PipelineProject {
    let envVars: BuildProjectEnvVars = {
        'REPOSITORY_URI': { value: props.ecrRepository.repositoryUri },
        'AWS_ACCOUNT_ID': { value: cdk.Aws.ACCOUNT_ID },
        'AWS_DEFAULT_REGION': { value: cdk.Aws.REGION },
        'CONTAINER_NAME': { value: props.containerName },
    };

    if (!fs.existsSync(props.buildSpecPath)) {
        throw new Error(`The buildspec.yml file does not exist at the specified path: ${props.buildSpecPath}`);
    }

    const buildSpecContent = yaml.parse(fs.readFileSync(props.buildSpecPath, 'utf8')) as { [key: string]: any };

    if (props.ecsService) {
        buildSpecContent.artifacts = {
            files: ['imagedefinitions.json']
        };
    }

    const codeBuildProject = new codebuild.PipelineProject(scope, `CodeBuildPipelineProject`, {
        projectName: `${props.pipelineName}Build`,
        environment: {
            privileged: true,
            buildImage: codebuild.LinuxBuildImage.STANDARD_7_0
        },
        buildSpec: codebuild.BuildSpec.fromObject(buildSpecContent),
        environmentVariables: envVars
    });

    props.ecrRepository.grantPullPush(codeBuildProject.role!);

    return codeBuildProject;
}
