variable "region" {
  type    = string
  default = "us-east-1"
}

variable "accounts" {
  description = "Account ids for each role in the landing zone"
  type = object({
    shared   = string
    dev      = string
    prod     = string
    security = string
  })
}

variable "bootstrap_role" {
  description = "Role Terraform assumes in each member account (created by Organizations / Control Tower)"
  type        = string
  default     = "OrganizationAccountAccessRole"
}

variable "external_id" {
  description = "Shared secret required on every cross-account AssumeRole (confused-deputy protection)"
  type        = string
  sensitive   = true
}

variable "workloads_ou_id" {
  type = string
}

variable "allowed_regions" {
  type    = list(string)
  default = ["us-east-1", "us-west-2"]
}

variable "approved_models" {
  type = list(string)
  default = [
    "anthropic.claude-sonnet-4-5*", "us.anthropic.claude-sonnet-4-5*",
    "anthropic.claude-haiku-4-5*", "us.anthropic.claude-haiku-4-5*",
    "amazon.titan-embed-text-v2*", "cohere.embed-english-v3*"
  ]
}
