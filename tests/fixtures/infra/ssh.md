# SSH Configuration and Security

## Key-Based Authentication

Password authentication is insecure for production servers. Always use SSH keys:

```bash
# Generate an Ed25519 key (preferred over RSA)
ssh-keygen -t ed25519 -C "admin@example.com"

# Copy public key to remote server
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@server.example.com
```

Disable password authentication in `/etc/ssh/sshd_config`:

```
PasswordAuthentication no
PubkeyAuthentication yes
PermitRootLogin prohibit-password
```

After changing sshd_config, reload the service: `sudo systemctl reload sshd`.

## SSH Config File

The `~/.ssh/config` file simplifies connections to frequently used hosts:

```
Host prod-web
    HostName 10.0.1.50
    User deploy
    IdentityFile ~/.ssh/deploy_ed25519
    Port 2222
    ForwardAgent no

Host staging-*
    HostName %h.staging.internal
    User admin
    ProxyJump bastion

Host bastion
    HostName bastion.example.com
    User jump
    IdentityFile ~/.ssh/bastion_key
```

Use `ProxyJump` instead of deprecated `ProxyCommand` for jump hosts. The `%h` token expands to the target hostname.

## SSH Tunnels

SSH tunnels encrypt traffic for services that do not natively support encryption:

```bash
# Local port forward — access remote PostgreSQL locally
ssh -L 5432:db.internal:5432 bastion-host

# Remote port forward — expose local dev server to remote
ssh -R 8080:localhost:3000 remote-host

# Dynamic SOCKS proxy
ssh -D 1080 proxy-host
```

Local forwarding (`-L`) maps a local port to a remote destination through the SSH tunnel. Remote forwarding (`-R`) does the reverse. Dynamic forwarding (`-D`) creates a SOCKS proxy.

## SSH Hardening

Security best practices for SSH servers:

- Change default port: `Port 2222`
- Limit users: `AllowUsers deploy admin`
- Set idle timeout: `ClientAliveInterval 300` and `ClientAliveCountMax 2`
- Disable X11 forwarding: `X11Forwarding no`
- Use `fail2ban` to block brute-force attempts
