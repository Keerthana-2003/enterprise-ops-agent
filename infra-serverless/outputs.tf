output "api_url" {
  value = aws_apigatewayv2_api.http.api_endpoint
}

output "ecr_repository" {
  value = aws_ecr_repository.agent.repository_url
}

output "event_bus" {
  value = aws_cloudwatch_event_bus.ops.name
}

output "results_bucket" {
  value = aws_s3_bucket.results.bucket
}

output "kms_key_arn" {
  value = aws_kms_key.app.arn
}
