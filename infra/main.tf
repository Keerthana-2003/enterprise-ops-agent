# Deploys the agent API to AWS: container image in ECR, service on App Runner,
# OpenAI key in Secrets Manager, health checks on /health.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ------------------------------------------------------------------ image registry

resource "aws_ecr_repository" "agent" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "agent" {
  repository = aws_ecr_repository.agent.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

# ------------------------------------------------------------------ secrets

resource "aws_secretsmanager_secret" "openai" {
  name                    = "${var.name}/openai-api-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "openai" {
  secret_id     = aws_secretsmanager_secret.openai.id
  secret_string = var.openai_api_key
}

# ------------------------------------------------------------------ IAM

# Lets App Runner pull the image from ECR.
resource "aws_iam_role" "apprunner_ecr" {
  name = "${var.name}-apprunner-ecr"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "build.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# Identity of the running container: may read only its own secret.
resource "aws_iam_role" "apprunner_instance" {
  name = "${var.name}-apprunner-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "tasks.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "bedrock" {
  name = "invoke-bedrock"
  role = aws_iam_role.apprunner_instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:ApplyGuardrail", "bedrock:Retrieve"]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy" "read_secret" {
  name = "read-openai-secret"
  role = aws_iam_role.apprunner_instance.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.openai.arn
    }]
  })
}

# ------------------------------------------------------------------ service

resource "aws_apprunner_service" "agent" {
  service_name = var.name

  source_configuration {
    auto_deployments_enabled = true

    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.agent.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          AGENT_MODEL          = var.agent_model
          LLM_PROVIDER         = var.llm_provider
          BEDROCK_MODEL_ID     = var.bedrock_model_id
          AGENT_RUNTIME        = var.agent_runtime
          COGNITO_USER_POOL_ID = aws_cognito_user_pool.users.id
          COGNITO_CLIENT_ID    = aws_cognito_user_pool_client.api.id
          COGNITO_REGION       = var.region
          BEDROCK_GUARDRAIL_ID      = aws_bedrock_guardrail.agent.guardrail_id
          BEDROCK_GUARDRAIL_VERSION = aws_bedrock_guardrail_version.agent.version
        }
        runtime_environment_secrets = {
          OPENAI_API_KEY = aws_secretsmanager_secret.openai.arn
        }
      }
    }
  }

  instance_configuration {
    cpu               = var.cpu
    memory            = var.memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 10
    healthy_threshold   = 1
    unhealthy_threshold = 3
  }

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr, aws_iam_role_policy.read_secret]
}
