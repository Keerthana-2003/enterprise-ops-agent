variable "environment" {
  type = string
}

variable "ci_account_id" {
  type = string
}

variable "security_account_id" {
  type = string
}

variable "external_id" {
  type      = string
  sensitive = true
}

variable "allowed_bedrock_model" {
  type = list(string)
}

variable "require_deploy_tag" {
  type    = bool
  default = false
}
