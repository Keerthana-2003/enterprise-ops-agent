"""OpenTelemetry tracing (vendor-neutral), exportable to Datadog or any OTLP backend.

The agent always creates spans through the OpenTelemetry API, which is a no-op until a tracer
provider is configured. `configure_otel()` installs the SDK with an OTLP/HTTP exporter when
OTEL_EXPORTER_OTLP_ENDPOINT is set, e.g. http://datadog-agent:4318 for the Datadog Agent's OTLP
receiver (see deploy/docker-compose.datadog.yml). Span attributes follow the OpenTelemetry
GenAI semantic conventions (gen_ai.*).
"""
from __future__ import annotations

import logging
import os

from opentelemetry import trace

log = logging.getLogger(__name__)
tracer = trace.get_tracer("enterprise-ops-agent")
_configured = False


def configure_otel(service_name: str = "enterprise-ops-agent") -> bool:
    """Install the OTel SDK with an OTLP exporter if an endpoint is configured. Idempotent."""
    global _configured
    if _configured or not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return _configured
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
            "deployment.environment": os.getenv("DEPLOY_ENV", "dev"),
        }))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _configured = True
    except Exception as err:  # SDK missing or misconfigured: keep running without export
        log.warning("OpenTelemetry export disabled: %s", err)
    return _configured
