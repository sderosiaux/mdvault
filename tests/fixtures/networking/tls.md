# TLS/SSL Certificate Management

## Certificate Types

- **Domain Validated (DV)**: Proves domain ownership. Automated via ACME/Let's Encrypt.
- **Organization Validated (OV)**: Verifies organization identity. Manual process.
- **Extended Validation (EV)**: Strict identity verification. Green bar in older browsers.

For most use cases, DV certificates from Let's Encrypt are sufficient and free.

## Let's Encrypt with Certbot

Automated certificate issuance and renewal:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate (Nginx plugin)
sudo certbot --nginx -d example.com -d www.example.com

# Obtain certificate (standalone)
sudo certbot certonly --standalone -d example.com

# Test renewal
sudo certbot renew --dry-run

# Auto-renewal via systemd timer (usually installed automatically)
sudo systemctl status certbot.timer
```

Certbot stores certificates in `/etc/letsencrypt/live/domain/`. Files: `fullchain.pem` (cert + intermediate), `privkey.pem` (private key).

## TLS Best Practices

Modern TLS configuration for 2024:

- Minimum TLS 1.2, prefer TLS 1.3
- Disable SSLv3, TLS 1.0, TLS 1.1
- Use AEAD ciphers (AES-GCM, ChaCha20-Poly1305)
- Enable HSTS with long max-age
- Enable OCSP stapling

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
ssl_prefer_server_ciphers off;
add_header Strict-Transport-Security "max-age=63072000" always;
ssl_stapling on;
ssl_stapling_verify on;
```

## Wildcard Certificates

A single certificate covering all subdomains:

```bash
certbot certonly --dns-cloudflare \
    --dns-cloudflare-credentials ~/.secrets/cloudflare.ini \
    -d "*.example.com" -d example.com
```

Wildcard certificates require DNS-01 challenge validation, not HTTP-01.
