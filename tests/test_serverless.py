"""Event-driven Lambda consumer: partial batch failures, idempotency, S3 + EventBridge outputs."""
from __future__ import annotations

import json

import serverless.handlers as h
from serverless.handlers import EventProcessor


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)

    def put_object(self, Bucket, Key, Body, **kw):
        assert kw["ServerSideEncryption"] == "aws:kms"
        self.objects[Key] = json.loads(Body)


class FakeEvents:
    def __init__(self):
        self.entries = []

    def put_events(self, Entries):
        self.entries += Entries


def sqs(msg_id, body):
    return {"messageId": msg_id, "body": json.dumps(body)}


def test_partial_batch_failure_idempotency_and_outputs():
    h._processed.clear()
    calls = []

    def runner(task):
        calls.append(task)
        if "boom" in task:
            raise RuntimeError("LLM timeout")
        return {"status": "completed", "proposed_actions": [{"action_type": "escalate"}]}

    s3, ev = FakeS3(), FakeEvents()
    p = EventProcessor(runner, s3, ev, bucket="results", bus="ops")
    event = {"Records": [
        sqs("m1", {"id": "evt-1", "detail-type": "TicketEscalated", "detail": {"task": "Review Fabrikam P1"}}),
        sqs("m2", {"id": "evt-2", "detail": {"task": "boom"}}),
        sqs("m3", {"id": "evt-3", "detail": {}}),
        sqs("m4", {"task": "raw message without envelope"}),
    ]}
    out = p.handle(event)
    assert out == {"batchItemFailures": [{"itemIdentifier": "m2"}, {"itemIdentifier": "m3"}]}
    assert set(s3.objects) == {"runs/evt-1.json", "runs/m4.json"}
    assert [json.loads(e["Detail"])["event_id"] for e in ev.entries] == ["evt-1", "m4"]

    # redelivery of an already processed event does not run the agent again (also across cold starts via S3)
    h._processed.clear()
    p.handle({"Records": [sqs("m5", {"id": "evt-1", "detail": {"task": "Review Fabrikam P1"}})]})
    assert calls.count("Review Fabrikam P1") == 1
