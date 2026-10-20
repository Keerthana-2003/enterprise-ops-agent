output "ecr_repository_url" {
  value = aws_ecr_repository.agent.repository_url
}

output "service_url" {
  value = "https://${aws_apprunner_service.agent.service_url}"
}

output "cloudfront_url" {
  value = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.users.id
}

output "cognito_client_id" {
  value = aws_cognito_user_pool_client.api.id
}
