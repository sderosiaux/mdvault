# Load Balancing Strategies

## Layer 4 vs Layer 7

**Layer 4 (Transport)**: Routes based on IP and TCP/UDP port. Fast but no content inspection. Examples: HAProxy TCP mode, AWS NLB.

**Layer 7 (Application)**: Routes based on HTTP headers, URL paths, cookies. More flexible but higher overhead. Examples: HAProxy HTTP mode, AWS ALB, Nginx.

## HAProxy Configuration

```
frontend http_front
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/combined.pem
    redirect scheme https if !{ ssl_fc }
    default_backend app_servers

backend app_servers
    balance roundrobin
    option httpchk GET /healthz
    http-check expect status 200
    server app1 10.0.0.1:8080 check weight 3
    server app2 10.0.0.2:8080 check weight 1
    server app3 10.0.0.3:8080 check backup
```

## Health Checks

Always configure health checks to avoid routing traffic to unhealthy backends. Health check endpoints should verify actual service readiness, not just return 200.
