# Guardrails for Amazon Bedrock: applied to every Converse call and to user input / final answers.

resource "aws_bedrock_guardrail" "agent" {
  name                      = "${var.name}-guardrail"
  description               = "Content, prompt-attack, PII and topic controls for the ops agent"
  blocked_input_messaging   = "This request can't be processed. Please rephrase it as an operations question."
  blocked_outputs_messaging = "The answer was withheld by the content policy. A human reviewer will follow up."

  content_policy_config {
    filters_config {
      type            = "HATE"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "INSULTS"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "MISCONDUCT"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "PROMPT_ATTACK"
      input_strength  = "HIGH"
      output_strength = "NONE"
    }
  }

  sensitive_information_policy_config {
    pii_entities_config {
      type   = "EMAIL"
      action = "ANONYMIZE"
    }
    pii_entities_config {
      type   = "PHONE"
      action = "ANONYMIZE"
    }
    pii_entities_config {
      type   = "CREDIT_DEBIT_CARD_NUMBER"
      action = "BLOCK"
    }
  }

  topic_policy_config {
    topics_config {
      name       = "legal_or_investment_advice"
      type       = "DENY"
      definition = "Requests for legal opinions or investment recommendations, which the agent must not give."
      examples   = ["Should we sue this customer?", "Which stocks should the company buy with this cash?"]
    }
  }
}

resource "aws_bedrock_guardrail_version" "agent" {
  guardrail_arn = aws_bedrock_guardrail.agent.guardrail_arn
  description   = "Initial published version"
}

output "bedrock_guardrail_id" {
  value = aws_bedrock_guardrail.agent.guardrail_id
}
