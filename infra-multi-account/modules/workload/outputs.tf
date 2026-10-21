output "deploy_role_arn" {
  value = aws_iam_role.deploy.arn
}

output "runtime_role_arn" {
  value = aws_iam_role.runtime.arn
}

output "audit_role_arn" {
  value = aws_iam_role.audit.arn
}

output "permission_boundary_arn" {
  value = aws_iam_policy.boundary.arn
}
