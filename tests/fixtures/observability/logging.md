# Centralized Logging

## ELK Stack Overview

The ELK stack (Elasticsearch, Logstash, Kibana) is a popular centralized logging solution. Modern deployments often replace Logstash with Filebeat or Fluentd for log shipping.

## Structured Logging

Always use structured logging in applications. JSON format enables efficient parsing and querying:

```python
import structlog

logger = structlog.get_logger()

logger.info("request_processed",
    method="POST",
    path="/api/users",
    status=201,
    duration_ms=45.2,
    user_id="u-12345"
)
```

Output:
```json
{"event": "request_processed", "method": "POST", "path": "/api/users", "status": 201, "duration_ms": 45.2, "user_id": "u-12345", "timestamp": "2024-01-15T10:30:00Z"}
```

## Fluentd Configuration

Fluentd collects, transforms, and forwards logs:

```xml
<source>
  @type tail
  path /var/log/app/*.log
  pos_file /var/log/fluentd/app.pos
  tag app.logs
  <parse>
    @type json
  </parse>
</source>

<filter app.logs>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    environment production
  </record>
</filter>

<match app.logs>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name app-logs
  <buffer>
    @type memory
    flush_interval 5s
  </buffer>
</match>
```

## Log Levels

Use appropriate log levels consistently:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational events
- **WARNING**: Unexpected but recoverable situations
- **ERROR**: Failures requiring attention
- **CRITICAL**: System-level failures

In production, set minimum level to INFO. Enable DEBUG only for troubleshooting.
