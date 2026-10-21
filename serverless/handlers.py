"""AWS Lambda handlers (container image from ECR; see infra-serverless/).

api_handler     API Gateway HTTP API -> the FastAPI app through Mangum (same endpoints as App Runner).
events_handler  SQS (fed by an EventBridge rule) -> one agent run per event. Uses partial batch
                responses so only failed messages are retried; poison messages go to the DLQ after
                maxReceiveCount. Runs are idempotent on the event id. Results are written to S3
                (SSE-KMS) and a completion event is published back to EventBridge.
Secrets: the LLM API key is read once from Secrets Manager (KMS-encrypted) at cold start.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

log = logging.getLogger("serverless")
_processed: set[str] = set()  # warm-container idempotency; S3 object existence covers cold starts


def _load_secret_env() -> None:
    arn = os.getenv("LLM_API_KEY_SECRET_ARN")
    if arn and not os.getenv("OPENAI_API_KEY"):
        import boto3

        os.environ["OPENAI_API_KEY"] = boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]


def api_handler(event, context):
    global _mangum
    if "_mangum" not in globals():
        from mangum import Mangum

        _load_secret_env()
        from api.main import app

        _mangum = Mangum(app, lifespan="off")
    return _mangum(event, context)


def _default_runner(task: str) -> dict:
    from agent.core import Agent
    from agent.providers import make_llm
    from agent.seed import DATA, seed

    if not (DATA / "crm.db").exists():
        seed(DATA)
    return asyncio.run(Agent(make_llm(), DATA).arun(task)).to_dict()


class EventProcessor:
    def __init__(self, runner=_default_runner, s3=None, events=None, bucket: str | None = None,
                 bus: str | None = None):
        self.runner, self.s3, self.events = runner, s3, events
        self.bucket, self.bus = bucket or os.getenv("RESULTS_BUCKET"), bus or os.getenv("EVENT_BUS_NAME")

    def _already_done(self, key: str) -> bool:
        if key in _processed:
            return True
        if self.s3 and self.bucket:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False
        return False

    def process_record(self, record: dict) -> None:
        body = json.loads(record["body"])
        event = body if "detail" in body else {"id": record["messageId"], "detail": body}  # EventBridge envelope or raw
        detail = event["detail"]
        task = detail.get("task") or detail.get("description")
        if not task:
            raise ValueError("event detail needs 'task'")
        key = f"runs/{event['id']}.json"
        if self._already_done(key):
            return
        result = self.runner(task)
        if self.s3 and self.bucket:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(result, default=str).encode(),
                               ServerSideEncryption="aws:kms", ContentType="application/json")
        if self.events and self.bus:
            self.events.put_events(Entries=[{"EventBusName": self.bus, "Source": "ops-agent", "DetailType": "AgentRunCompleted",
                                             "Detail": json.dumps({"event_id": event["id"], "status": result.get("status"),
                                                                   "proposed_actions": result.get("proposed_actions", []),
                                                                   "result_key": key})}])
        _processed.add(key)

    def handle(self, event: dict) -> dict:
        failures = []
        for record in event.get("Records", []):
            try:
                self.process_record(record)
            except Exception as err:  # report only this message as failed; the rest of the batch is deleted
                log.warning("record %s failed: %s", record.get("messageId"), err)
                failures.append({"itemIdentifier": record["messageId"]})
        return {"batchItemFailures": failures}


def events_handler(event, context):
    global _processor
    if "_processor" not in globals():
        import boto3

        _load_secret_env()
        _processor = EventProcessor(s3=boto3.client("s3"), events=boto3.client("events"))
    return _processor.handle(event)
