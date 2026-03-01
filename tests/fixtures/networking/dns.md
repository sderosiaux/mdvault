# DNS Configuration and Troubleshooting

## Record Types

Common DNS record types and their uses:

| Record | Purpose | Example |
|--------|---------|---------|
| A | IPv4 address | `example.com → 93.184.216.34` |
| AAAA | IPv6 address | `example.com → 2606:2800:220:1::` |
| CNAME | Alias to another domain | `www.example.com → example.com` |
| MX | Mail exchange server | `example.com → mail.example.com (pri 10)` |
| TXT | Arbitrary text (SPF, DKIM) | `v=spf1 include:_spf.google.com ~all` |
| NS | Nameserver delegation | `example.com → ns1.provider.com` |
| SRV | Service location | `_sip._tcp.example.com → sip.example.com:5060` |

## Debugging DNS Issues

Essential commands for DNS troubleshooting:

```bash
# Query specific record type
dig example.com A +short
dig example.com MX +short

# Trace full resolution path
dig example.com +trace

# Query specific nameserver
dig @8.8.8.8 example.com

# Reverse lookup
dig -x 93.184.216.34

# Check DNS propagation
dig example.com @ns1.provider.com
dig example.com @8.8.4.4
```

## TTL Management

Time-to-live controls how long resolvers cache records. Before making DNS changes:

1. Lower TTL to 60 seconds at least 48 hours before the change
2. Make the DNS change
3. Wait for old TTL to expire
4. Verify propagation with `dig` against multiple resolvers
5. Raise TTL back to normal (3600-86400 seconds)

## Split-Horizon DNS

Different responses based on the source network. Useful for internal vs external services:

```bind
view "internal" {
    match-clients { 10.0.0.0/8; };
    zone "example.com" {
        type master;
        file "/etc/bind/internal.example.com";
    };
};

view "external" {
    match-clients { any; };
    zone "example.com" {
        type master;
        file "/etc/bind/external.example.com";
    };
};
```
