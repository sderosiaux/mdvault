# Container Security

## Image Scanning

Scan container images for vulnerabilities before deployment:

```bash
# Using Trivy
trivy image myapp:latest
trivy image --severity HIGH,CRITICAL myapp:latest

# Using Grype
grype myapp:latest
```

## Minimal Base Images

Use distroless or Alpine-based images to reduce attack surface:

```dockerfile
# Distroless (no shell, no package manager)
FROM gcr.io/distroless/python3-debian12
COPY --from=builder /app /app
CMD ["app/main.py"]

# Alpine (minimal but has shell)
FROM python:3.12-alpine
```

## Non-Root Containers

Never run containers as root:

```dockerfile
RUN addgroup -g 1001 appgroup && adduser -u 1001 -G appgroup -D appuser
USER appuser
```

## Read-Only Filesystem

Mount the root filesystem as read-only:

```yaml
services:
  app:
    read_only: true
    tmpfs:
      - /tmp
```
