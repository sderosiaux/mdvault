# Firewall Configuration

## UFW (Uncomplicated Firewall)

UFW is the default firewall tool on Ubuntu/Debian:

```bash
# Enable firewall
sudo ufw enable

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific services
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow from specific IP
sudo ufw allow from 10.0.0.0/24 to any port 5432

# Delete a rule
sudo ufw delete allow 80/tcp

# Check status
sudo ufw status verbose
```

## iptables

Lower-level firewall management for advanced use cases:

```bash
# Drop all incoming by default
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow SSH
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Rate limit SSH connections
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 3/min --limit-burst 3 -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: " --log-level 4

# Save rules
iptables-save > /etc/iptables/rules.v4
```

## Network Security Groups (Cloud)

AWS Security Group example for a web application:

```
Inbound:
  - Port 443 (HTTPS) from 0.0.0.0/0
  - Port 80 (HTTP) from 0.0.0.0/0 (redirect to HTTPS)
  - Port 22 (SSH) from 10.0.0.0/8 (VPN only)

Outbound:
  - All traffic to 0.0.0.0/0
```

Principle of least privilege: only open ports that are actively needed. Review security groups quarterly.
