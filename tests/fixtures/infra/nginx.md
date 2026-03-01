# Nginx Configuration Guide

## Reverse Proxy Setup

Nginx is widely used as a reverse proxy to forward client requests to backend services. A basic reverse proxy configuration looks like this:

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

The `proxy_set_header` directives ensure the backend receives the original client information rather than the proxy's details. This is crucial for logging and security.

## SSL/TLS Termination

SSL termination at the Nginx layer offloads encryption from backend services:

```nginx
server {
    listen 443 ssl http2;
    server_name secure.example.com;

    ssl_certificate /etc/ssl/certs/example.crt;
    ssl_certificate_key /etc/ssl/private/example.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://backend;
    }
}
```

Always prefer TLSv1.2+ and disable weak ciphers. Use `ssl_session_cache` to reduce handshake overhead for repeat connections.

## Upstream Load Balancing

Define upstream blocks for distributing traffic across multiple backends:

```nginx
upstream backend {
    least_conn;
    server 10.0.0.1:8080 weight=3;
    server 10.0.0.2:8080 weight=1;
    server 10.0.0.3:8080 backup;
}
```

Load balancing methods include `round-robin` (default), `least_conn`, and `ip_hash`. The `backup` directive marks a server that only receives traffic when all primary servers are down.

## Rate Limiting

Protect your services from abuse with rate limiting:

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://backend;
    }
}
```

The `burst` parameter allows temporary spikes, while `nodelay` processes burst requests immediately rather than queuing them.

## Caching Configuration

Enable proxy caching for improved performance:

```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m max_size=1g inactive=60m;

server {
    location / {
        proxy_cache my_cache;
        proxy_cache_valid 200 302 10m;
        proxy_cache_valid 404 1m;
        add_header X-Cache-Status $upstream_cache_status;
        proxy_pass http://backend;
    }
}
```

Monitor cache hit rates using the `X-Cache-Status` header. Values include HIT, MISS, BYPASS, and EXPIRED.

## Gzip Compression

Enable compression to reduce bandwidth:

```nginx
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css application/json application/javascript text/xml;
```
