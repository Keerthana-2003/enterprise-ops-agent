# ADR-0002: RAG instead of fine-tuning for policy knowledge

**Status:** accepted

**Context.** Answers must follow escalation, credit and collections policies that change during
the year and must be cited for audit.

**Decision.** Retrieve policy passages at query time (pgvector, or Bedrock Knowledge Bases when the
client prefers a managed service) and require the model to cite them in a structured answer.

**Consequences.** Policy updates take effect on re-ingestion, with no retraining. Citations make
answers auditable. Retrieval quality becomes a measured component (Recall@K in the eval set).
Fine-tuning stays an option for output format or tone, not for knowledge.
