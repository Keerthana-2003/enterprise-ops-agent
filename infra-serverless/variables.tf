variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "ops-agent"
}

variable "image_tag" {
  type    = string
  default = "v1"
}

variable "llm_provider" {
  type    = string
  default = "bedrock"
}

variable "bedrock_model_id" {
  type    = string
  default = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "cognito_user_pool_id" {
  type = string
}

variable "cognito_client_id" {
  type = string
}

variable "allowed_origins" {
  type    = list(string)
  default = ["https://localhost:3000"]
}

variable "worker_concurrency" {
  type    = number
  default = 5
}
