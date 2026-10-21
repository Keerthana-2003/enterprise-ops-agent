# Serverless + event-driven deployment of the agent:
#   API Gateway HTTP API (Cognito JWT authorizer) -> Lambda (container image in ECR, FastAPI via Mangum)
#   EventBridge rule -> SQS (DLQ, KMS) -> Lambda consumer (partial batch failures) -> results in S3 + completion events
# One customer-managed KMS key encrypts the secret, logs, queues, bucket and Lambda environment.
# Least-privilege IAM: each function role can reach only its own resources and the Bedrock models it uses.

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
  default_tags {
    tags = { app = var.name, managed_by = "terraform" }
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  image_uri  = "${aws_ecr_repository.agent.repository_url}:${var.image_tag}"
  env = {
    LLM_PROVIDER           = var.llm_provider
    BEDROCK_MODEL_ID       = var.bedrock_model_id
    LLM_API_KEY_SECRET_ARN = aws_secretsmanager_secret.llm_key.arn
    RESULTS_BUCKET         = aws_s3_bucket.results.bucket
    EVENT_BUS_NAME         = aws_cloudwatch_event_bus.ops.name
  }
}

# ------------------------------------------------------------------ KMS

data "aws_iam_policy_document" "kms" {
  statement {
    sid       = "AccountAdmin"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:${data.aws_partition.current.partition}:iam::${local.account_id}:root"]
    }
  }
  statement {
    sid       = "CloudWatchLogs"
    actions   = ["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["logs.${var.region}.amazonaws.com"]
    }
    condition {
      test     = "ArnLike"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:${data.aws_partition.current.partition}:logs:${var.region}:${local.account_id}:log-group:/aws/lambda/${var.name}-*"]
    }
  }
  statement {
    sid       = "EventBridgeToSqs"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_kms_key" "app" {
  description             = "${var.name} secrets, logs, queues and results"
  enable_key_rotation     = true
  deletion_window_in_days = 7
  policy                  = data.aws_iam_policy_document.kms.json
}

resource "aws_kms_alias" "app" {
  name          = "alias/${var.name}"
  target_key_id = aws_kms_key.app.key_id
}

# ------------------------------------------------------------------ ECR, secret, bucket

resource "aws_ecr_repository" "agent" {
  name                 = "${var.name}-lambda"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.app.arn
  }
}

resource "aws_secretsmanager_secret" "llm_key" {
  name                    = "${var.name}/llm-api-key"
  kms_key_id              = aws_kms_key.app.arn
  recovery_window_in_days = 0
}

resource "aws_s3_bucket" "results" {
  bucket_prefix = "${var.name}-results-"
  force_destroy = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.app.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "results" {
  bucket = aws_s3_bucket.results.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ------------------------------------------------------------------ IAM (one role per function)

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "common" {
  statement {
    sid       = "ReadLlmKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.llm_key.arn]
  }
  statement {
    sid       = "UseKey"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.app.arn]
  }
  statement {
    sid       = "InvokeBedrock"
    actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:Converse", "bedrock:ConverseStream", "bedrock:ApplyGuardrail"]
    resources = ["arn:${data.aws_partition.current.partition}:bedrock:*::foundation-model/*", "arn:${data.aws_partition.current.partition}:bedrock:${var.region}:${local.account_id}:*"]
  }
}

data "aws_iam_policy_document" "worker" {
  source_policy_documents = [data.aws_iam_policy_document.common.json]
  statement {
    sid       = "ConsumeQueue"
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"]
    resources = [aws_sqs_queue.tasks.arn]
  }
  statement {
    sid       = "WriteResults"
    actions   = ["s3:PutObject", "s3:GetObject"]
    resources = ["${aws_s3_bucket.results.arn}/runs/*"]
  }
  statement {
    sid       = "ListResults"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.results.arn]
  }
  statement {
    sid       = "PublishCompletion"
    actions   = ["events:PutEvents"]
    resources = [aws_cloudwatch_event_bus.ops.arn]
  }
}

resource "aws_iam_role" "api" {
  name               = "${var.name}-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role" "worker" {
  name               = "${var.name}-worker"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy" "api" {
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.common.json
}

resource "aws_iam_role_policy" "worker" {
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker.json
}

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "worker_logs" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ------------------------------------------------------------------ Lambda functions

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${var.name}-api"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.app.arn
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/aws/lambda/${var.name}-worker"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.app.arn
}

resource "aws_lambda_function" "api" {
  function_name = "${var.name}-api"
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = local.image_uri
  architectures = ["arm64"]
  memory_size   = 1024
  timeout       = 29
  kms_key_arn   = aws_kms_key.app.arn
  image_config {
    command = ["serverless.handlers.api_handler"]
  }
  environment {
    variables = local.env
  }
  tracing_config {
    mode = "Active"
  }
  depends_on = [aws_cloudwatch_log_group.api]
}

resource "aws_lambda_function" "worker" {
  function_name                  = "${var.name}-worker"
  role                           = aws_iam_role.worker.arn
  package_type                   = "Image"
  image_uri                      = local.image_uri
  architectures                  = ["arm64"]
  memory_size                    = 1024
  timeout                        = 300
  reserved_concurrent_executions = var.worker_concurrency
  kms_key_arn                    = aws_kms_key.app.arn
  image_config {
    command = ["serverless.handlers.events_handler"]
  }
  environment {
    variables = local.env
  }
  tracing_config {
    mode = "Active"
  }
  depends_on = [aws_cloudwatch_log_group.worker]
}

# ------------------------------------------------------------------ API Gateway (HTTP API + Cognito JWT)

resource "aws_apigatewayv2_api" "http" {
  name          = "${var.name}-http"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = var.allowed_origins
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["authorization", "content-type"]
  }
}

resource "aws_apigatewayv2_authorizer" "cognito" {
  api_id           = aws_apigatewayv2_api.http.id
  name             = "cognito"
  authorizer_type  = "JWT"
  identity_sources = ["$request.header.Authorization"]
  jwt_configuration {
    audience = [var.cognito_client_id]
    issuer   = "https://cognito-idp.${var.region}.amazonaws.com/${var.cognito_user_pool_id}"
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_route" "runs" {
  api_id             = aws_apigatewayv2_api.http.id
  route_key          = "ANY /runs/{proxy+}"
  target             = "integrations/${aws_apigatewayv2_integration.api.id}"
  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.cognito.id
}

resource "aws_apigatewayv2_route" "runs_root" {
  api_id             = aws_apigatewayv2_api.http.id
  route_key          = "POST /runs"
  target             = "integrations/${aws_apigatewayv2_integration.api.id}"
  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.cognito.id
}

resource "aws_cloudwatch_log_group" "access" {
  name              = "/aws/apigateway/${var.name}"
  retention_in_days = 30
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 50
    throttling_rate_limit  = 20
  }
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.access.arn
    format = jsonencode({
      requestId = "$context.requestId", route = "$context.routeKey", status = "$context.status",
      latencyMs = "$context.integrationLatency", user = "$context.authorizer.claims.sub"
    })
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# ------------------------------------------------------------------ Event-driven: EventBridge -> SQS -> Lambda

resource "aws_cloudwatch_event_bus" "ops" {
  name = "${var.name}-events"
}

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.name}-tasks-dlq"
  kms_master_key_id         = aws_kms_key.app.arn
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "tasks" {
  name                       = "${var.name}-tasks"
  kms_master_key_id          = aws_kms_key.app.arn
  visibility_timeout_seconds = 360 # > worker timeout
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })
}

data "aws_iam_policy_document" "queue" {
  statement {
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.tasks.arn]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_cloudwatch_event_rule.agent_tasks.arn]
    }
  }
}

resource "aws_sqs_queue_policy" "tasks" {
  queue_url = aws_sqs_queue.tasks.id
  policy    = data.aws_iam_policy_document.queue.json
}

# Business events that need an agent: P1 tickets, invoices going overdue, explicit requests.
resource "aws_cloudwatch_event_rule" "agent_tasks" {
  name           = "${var.name}-agent-tasks"
  event_bus_name = aws_cloudwatch_event_bus.ops.name
  event_pattern = jsonencode({
    source        = ["support", "billing", "ops-requests"]
    "detail-type" = ["TicketEscalated", "InvoiceOverdue", "AgentTaskRequested"]
  })
}

resource "aws_cloudwatch_event_target" "queue" {
  rule           = aws_cloudwatch_event_rule.agent_tasks.name
  event_bus_name = aws_cloudwatch_event_bus.ops.name
  arn            = aws_sqs_queue.tasks.arn
}

resource "aws_lambda_event_source_mapping" "worker" {
  event_source_arn        = aws_sqs_queue.tasks.arn
  function_name           = aws_lambda_function.worker.arn
  batch_size              = 5
  function_response_types = ["ReportBatchItemFailures"]
  scaling_config {
    maximum_concurrency = var.worker_concurrency
  }
}

resource "aws_cloudwatch_metric_alarm" "dlq" {
  alarm_name          = "${var.name}-dlq-not-empty"
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.dlq.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
}
