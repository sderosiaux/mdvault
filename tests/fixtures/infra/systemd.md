# Systemd Service Management

## Creating a Service Unit

Create a custom service file in `/etc/systemd/system/`:

```ini
[Unit]
Description=My Application
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=notify
User=appuser
Group=appgroup
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/venv/bin/python -m myapp.server
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5
TimeoutStartSec=30

Environment=APP_ENV=production
EnvironmentFile=/etc/myapp/env

[Install]
WantedBy=multi-user.target
```

## Common Commands

```bash
sudo systemctl daemon-reload
sudo systemctl enable myapp
sudo systemctl start myapp
sudo systemctl status myapp
journalctl -u myapp -f --since "10 minutes ago"
```

## Resource Control with Cgroups

Limit resources for a service:

```ini
[Service]
MemoryMax=512M
CPUQuota=50%
TasksMax=100
```
