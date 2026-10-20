from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_apprunner as apprunner
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parent.parent


class AgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, agent_model: str, llm_provider: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # Container image built from the repo's Dockerfile and pushed to the CDK asset ECR repo.
        image = ecr_assets.DockerImageAsset(
            self, "AgentImage", directory=str(REPO_ROOT), platform=ecr_assets.Platform.LINUX_AMD64,
            exclude=["infra", "infra-cdk", ".git", ".venv", "data", "evals/results"])

        # Set the real value after the first deploy:
        #   aws secretsmanager put-secret-value --secret-id <OpenAISecretArn> --secret-string sk-...
        openai_secret = secretsmanager.Secret(
            self, "OpenAIKey", description="OpenAI API key for the agent", removal_policy=RemovalPolicy.DESTROY)

        users = cognito.UserPool(
            self, "Users", self_sign_up_enabled=False, sign_in_aliases=cognito.SignInAliases(email=True),
            password_policy=cognito.PasswordPolicy(min_length=12, require_digits=True, require_symbols=True,
                                                   require_lowercase=True, require_uppercase=True),
            removal_policy=RemovalPolicy.DESTROY)
        client = users.add_client("ApiClient", generate_secret=False,
                                  auth_flows=cognito.AuthFlow(user_password=True, user_srp=True))

        access_role = iam.Role(self, "AppRunnerEcrAccess",
                               assumed_by=iam.ServicePrincipal("build.apprunner.amazonaws.com"))
        image.repository.grant_pull(access_role)

        instance_role = iam.Role(self, "AppRunnerInstance",
                                 assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"))
        openai_secret.grant_read(instance_role)
        instance_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], resources=["*"]))

        env = {
            "AGENT_MODEL": agent_model,
            "LLM_PROVIDER": llm_provider,
            "COGNITO_USER_POOL_ID": users.user_pool_id,
            "COGNITO_CLIENT_ID": client.user_pool_client_id,
            "COGNITO_REGION": self.region,
        }
        service = apprunner.CfnService(
            self, "Service",
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                auto_deployments_enabled=False,
                authentication_configuration=apprunner.CfnService.AuthenticationConfigurationProperty(
                    access_role_arn=access_role.role_arn),
                image_repository=apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=image.image_uri,
                    image_repository_type="ECR",
                    image_configuration=apprunner.CfnService.ImageConfigurationProperty(
                        port="8000",
                        runtime_environment_variables=[
                            apprunner.CfnService.KeyValuePairProperty(name=k, value=v) for k, v in env.items()],
                        runtime_environment_secrets=[
                            apprunner.CfnService.KeyValuePairProperty(
                                name="OPENAI_API_KEY", value=openai_secret.secret_arn)],
                    ),
                ),
            ),
            instance_configuration=apprunner.CfnService.InstanceConfigurationProperty(
                cpu="1 vCPU", memory="2 GB", instance_role_arn=instance_role.role_arn),
            health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
                protocol="HTTP", path="/health", interval=10, healthy_threshold=1, unhealthy_threshold=3),
        )

        cdn = cloudfront.Distribution(
            self, "Cdn",
            comment="Enterprise Ops Agent API",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.HttpOrigin(service.attr_service_url,
                                          protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                compress=False,
            ),
        )

        CfnOutput(self, "ServiceUrl", value=f"https://{service.attr_service_url}")
        CfnOutput(self, "CloudFrontUrl", value=f"https://{cdn.distribution_domain_name}")
        CfnOutput(self, "UserPoolId", value=users.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=client.user_pool_client_id)
        CfnOutput(self, "OpenAISecretArn", value=openai_secret.secret_arn)
