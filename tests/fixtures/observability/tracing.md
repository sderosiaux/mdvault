# Distributed Tracing

## OpenTelemetry Setup

OpenTelemetry is the standard for distributed tracing:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
```

## Instrumenting Code

```python
@tracer.start_as_current_span("process_order")
def process_order(order_id: str):
    span = trace.get_current_span()
    span.set_attribute("order.id", order_id)

    with tracer.start_as_current_span("validate_order"):
        validate(order_id)

    with tracer.start_as_current_span("charge_payment"):
        charge(order_id)
```

## Context Propagation

Traces span multiple services through context propagation headers. The W3C Trace Context standard uses `traceparent` and `tracestate` headers.

```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
```
