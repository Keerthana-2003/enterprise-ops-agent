#!/usr/bin/env python3
"""AWS CDK (Python) version of the deployment in infra/ (Terraform).

Same architecture: Docker image asset -> ECR, App Runner service, Secrets Manager key,
Bedrock access, Cognito user pool, CloudFront in front of the API.

  cd infra-cdk && pip install -r requirements.txt
  npx aws-cdk@2 bootstrap    # once per account/region
  npx aws-cdk@2 deploy
"""
import aws_cdk as cdk

from stack import AgentStack

app = cdk.App()
AgentStack(app, "EnterpriseOpsAgent",
           agent_model=app.node.try_get_context("agent_model") or "gpt-4o-mini",
           llm_provider=app.node.try_get_context("llm_provider") or "openai")
app.synth()
