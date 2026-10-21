# IAM for one workload account: permission boundary, CI deploy role (trusted from the shared-services
# account), the agent's runtime role, and a read-only audit role for the security account.

terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account = data.aws_caller_identity.current.account_id
  prefix  = "ops-agent-${var.environment}"
}

# Upper limit for every role the platform creates in this account. Deploy can only create roles
# that carry this same boundary, so privileges cannot be escalated by creating a new role.
resource "aws_iam_policy" "boundary" {
  name = "${local.prefix}-boundary"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid    = "AllowedServices", Effect = "Allow", Resource = "*",
        Action = ["lambda:*", "apigateway:*", "sqs:*", "events:*", "s3:*", "logs:*", "cloudwatch:*", "xray:*",
          "kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey*", "kms:DescribeKey", "kms:CreateGrant",
          "secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
        "ecr:GetAuthorizationToken", "bedrock:InvokeModel*", "bedrock:Converse*", "bedrock:ApplyGuardrail", "bedrock:Retrieve*", "bedrock-agentcore:*"]
      },
      { Sid    = "IamOnlyWithBoundary", Effect = "Allow", Resource = "arn:aws:iam::${local.account}:role/${local.prefix}-*",
        Action = ["iam:CreateRole", "iam:PutRolePolicy", "iam:AttachRolePolicy", "iam:DetachRolePolicy", "iam:DeleteRolePolicy"],
        Condition = { StringEquals = { "iam:PermissionsBoundary" = "arn:aws:iam::${local.account}:policy/${local.prefix}-boundary" } }
      },
      { Sid    = "IamReadAndPass", Effect = "Allow", Resource = "arn:aws:iam::${local.account}:role/${local.prefix}-*",
      Action = ["iam:GetRole", "iam:GetRolePolicy", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies", "iam:DeleteRole", "iam:TagRole", "iam:PassRole"] },
      { Sid    = "NoBoundaryTampering", Effect = "Deny", Resource = "*",
        Action = ["iam:DeleteRolePermissionsBoundary", "iam:PutRolePermissionsBoundary", "iam:CreatePolicyVersion", "iam:DeletePolicy", "iam:SetDefaultPolicyVersion"],
        Condition = { ArnLike = { "aws:PrincipalArn" = "arn:aws:iam::${local.account}:role/${local.prefix}-*" } }
      }
    ]
  })
}

# ------------------------------------------------------------------ CI/CD deploy role (trusted from shared services)

data "aws_iam_policy_document" "deploy_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${var.ci_account_id}:role/ops-agent-ci"]
    }
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.external_id]
    }
    dynamic "condition" {
      for_each = var.require_deploy_tag ? [1] : []
      content {
        test     = "StringEquals"
        variable = "aws:PrincipalTag/release-approved"
        values   = ["true"]
      }
    }
  }
}

resource "aws_iam_role" "deploy" {
  name                 = "${local.prefix}-deploy"
  assume_role_policy   = data.aws_iam_policy_document.deploy_trust.json
  permissions_boundary = aws_iam_policy.boundary.arn
  max_session_duration = 3600
}

resource "aws_iam_role_policy" "deploy" {
  role = aws_iam_role.deploy.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["lambda:*", "apigateway:*", "sqs:*", "events:*", "s3:*", "logs:*", "cloudwatch:*", "iam:*", "kms:*", "secretsmanager:*"], Resource = "*" }]
  })
  # effective permissions = this policy ∩ the boundary
}

# ------------------------------------------------------------------ agent runtime role (Lambda / AgentCore)

data "aws_iam_policy_document" "runtime_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com", "bedrock-agentcore.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account]
    }
  }
}

resource "aws_iam_role" "runtime" {
  name                 = "${local.prefix}-runtime"
  assume_role_policy   = data.aws_iam_policy_document.runtime_trust.json
  permissions_boundary = aws_iam_policy.boundary.arn
}

resource "aws_iam_role_policy" "runtime" {
  role = aws_iam_role.runtime.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect   = "Allow", Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream"],
        Resource = [for m in var.allowed_bedrock_model : "arn:aws:bedrock:*::foundation-model/${trimprefix(m, "us.")}"]
      },
      { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = "arn:aws:secretsmanager:${data.aws_region.current.name}:${local.account}:secret:ops-agent/*" },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:*:${local.account}:log-group:/aws/*/ops-agent*" }
    ]
  })
}

# ------------------------------------------------------------------ security audit role (read-only, MFA)

data "aws_iam_policy_document" "audit_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${var.security_account_id}:root"]
    }
    condition {
      test     = "Bool"
      variable = "aws:MultiFactorAuthPresent"
      values   = ["true"]
    }
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.external_id]
    }
  }
}

resource "aws_iam_role" "audit" {
  name               = "${local.prefix}-audit"
  assume_role_policy = data.aws_iam_policy_document.audit_trust.json
}

resource "aws_iam_role_policy_attachment" "audit" {
  role       = aws_iam_role.audit.name
  policy_arn = "arn:aws:iam::aws:policy/SecurityAudit"
}
