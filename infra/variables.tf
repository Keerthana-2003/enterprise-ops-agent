variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "enterprise-ops-agent"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "agent_model" {
  type    = string
  default = "gpt-4o-mini"
}

variable "openai_api_key" {
  type      = string
  sensitive = true
}

variable "cpu" {
  type    = string
  default = "1024"
}

variable "memory" {
  type    = string
  default = "2048"
}

variable "llm_provider" {
  type    = string
  default = "openai"
}

variable "bedrock_model_id" {
  type    = string
  default = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
}

variable "agent_runtime" {
  type    = string
  default = "loop"
}
