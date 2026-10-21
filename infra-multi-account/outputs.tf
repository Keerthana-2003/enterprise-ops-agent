output "ecr_repository_url" {
  value = aws_ecr_repository.agent.repository_url
}

output "deploy_role_arns" {
  value = { dev = module.dev.deploy_role_arn, prod = module.prod.deploy_role_arn }
}

output "audit_role_arns" {
  value = { dev = module.dev.audit_role_arn, prod = module.prod.audit_role_arn }
}

output "permission_boundary_arns" {
  value = { dev = module.dev.permission_boundary_arn, prod = module.prod.permission_boundary_arn }
}
