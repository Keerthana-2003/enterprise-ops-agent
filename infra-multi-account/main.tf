# Multi-account landing zone for the agent platform (AWS Organizations):
#
#   shared-services  ECR images, CI/CD deploy role, central KMS key for cross-account artifacts
#   workload (dev / prod)  the agent (serverless stack in ../infra-serverless), Bedrock access
#   security         audit role (read-only) assumed by the security team
#
# Terraform runs once from the management/CI account and assumes a role in each account
# (provider aliases). Trust relationships are explicit: every cross-account role names the exact
# principal it trusts, requires an ExternalId, and (for humans) MFA. Every role created for workloads
# carries a permission boundary, so even an over-broad inline policy cannot exceed it. An SCP
# denies leaving the org, disabling CloudTrail, using regions outside the allow-list, and invoking
# Bedrock models that are not approved.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source                = "hashicorp/aws"
      version               = "~> 5.0"
    }
  }
}

provider "aws" {
  alias  = "management"
  region = var.region
}

provider "aws" {
  alias  = "shared"
  region = var.region
  assume_role {
    role_arn     = "arn:aws:iam::${var.accounts.shared}:role/${var.bootstrap_role}"
    session_name = "terraform-shared"
    external_id  = var.external_id
  }
}

provider "aws" {
  alias  = "dev"
  region = var.region
  assume_role {
    role_arn     = "arn:aws:iam::${var.accounts.dev}:role/${var.bootstrap_role}"
    session_name = "terraform-dev"
    external_id  = var.external_id
  }
}

provider "aws" {
  alias  = "prod"
  region = var.region
  assume_role {
    role_arn     = "arn:aws:iam::${var.accounts.prod}:role/${var.bootstrap_role}"
    session_name = "terraform-prod"
    external_id  = var.external_id
  }
}

locals {
  workload_accounts = [var.accounts.dev, var.accounts.prod]
}

# ------------------------------------------------------------------ shared services: ECR + KMS shared to workloads

resource "aws_kms_key" "shared" {
  provider            = aws.shared
  description         = "Shared artifacts (ECR images) readable by workload accounts"
  enable_key_rotation = true
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid = "Admin", Effect = "Allow", Principal = { AWS = "arn:aws:iam::${var.accounts.shared}:root" }, Action = "kms:*", Resource = "*" },
      { Sid    = "WorkloadDecrypt", Effect = "Allow",
        Principal = { AWS = [for a in local.workload_accounts : "arn:aws:iam::${a}:root"] },
        Action    = ["kms:Decrypt", "kms:DescribeKey"], Resource = "*",
        Condition = { StringEquals = { "kms:ViaService" = "ecr.${var.region}.amazonaws.com" } }
      }
    ]
  })
}

resource "aws_ecr_repository" "agent" {
  provider             = aws.shared
  name                 = "ops-agent-lambda"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.shared.arn
  }
}

# Workload accounts' Lambda service may pull images (cross-account ECR for Lambda container images).
resource "aws_ecr_repository_policy" "agent" {
  provider   = aws.shared
  repository = aws_ecr_repository.agent.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid = "WorkloadPull", Effect = "Allow",
        Principal = { AWS = [for a in local.workload_accounts : "arn:aws:iam::${a}:root"] },
      Action = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"] },
      { Sid = "LambdaPull", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" },
        Action    = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
        Condition = { StringLike = { "aws:sourceArn" = [for a in local.workload_accounts : "arn:aws:lambda:${var.region}:${a}:function:*"] } }
      }
    ]
  })
}

# ------------------------------------------------------------------ per-workload-account IAM (module)

module "dev" {
  source = "./modules/workload"
  providers = {
    aws = aws.dev
  }
  environment           = "dev"
  ci_account_id         = var.accounts.shared
  security_account_id   = var.accounts.security
  external_id           = var.external_id
  allowed_bedrock_model = var.approved_models
}

module "prod" {
  source = "./modules/workload"
  providers = {
    aws = aws.prod
  }
  environment           = "prod"
  ci_account_id         = var.accounts.shared
  security_account_id   = var.accounts.security
  external_id           = var.external_id
  allowed_bedrock_model = var.approved_models
  require_deploy_tag    = true
}

# ------------------------------------------------------------------ guardrails for the whole workloads OU (SCP)

resource "aws_organizations_policy" "workload_guardrails" {
  provider    = aws.management
  name        = "ai-workload-guardrails"
  description = "Region allow-list, protect CloudTrail and org membership, approved Bedrock models only"
  type        = "SERVICE_CONTROL_POLICY"
  content = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid = "DenyLeaveOrgAndTrailTampering", Effect = "Deny", Resource = "*",
      Action = ["organizations:LeaveOrganization", "cloudtrail:StopLogging", "cloudtrail:DeleteTrail", "cloudtrail:UpdateTrail"] },
      { Sid    = "DenyOutsideApprovedRegions", Effect = "Deny", Resource = "*",
        NotAction = ["iam:*", "organizations:*", "sts:*", "support:*", "cloudfront:*", "route53:*", "budgets:*"],
        Condition = { StringNotEquals = { "aws:RequestedRegion" = var.allowed_regions } }
      },
      { Sid      = "DenyUnapprovedModels", Effect = "Deny",
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream"],
        Resource = "*",
        Condition = { "ForAllValues:StringNotLike" = { "bedrock:ModelId" = var.approved_models },
        "Null" = { "bedrock:ModelId" = "false" } }
      }
    ]
  })
}

resource "aws_organizations_policy_attachment" "workloads" {
  provider  = aws.management
  policy_id = aws_organizations_policy.workload_guardrails.id
  target_id = var.workloads_ou_id
}
