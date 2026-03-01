# Secrets Management

## HashiCorp Vault

Vault is the industry standard for secrets management. It provides dynamic secrets, encryption as a service, and detailed audit logging.

## Static Secrets (KV Engine)

```bash
# Enable KV v2 engine
vault secrets enable -path=secret kv-v2

# Store a secret
vault kv put secret/myapp/db username=admin password=s3cur3p4ss

# Retrieve a secret
vault kv get secret/myapp/db
vault kv get -field=password secret/myapp/db

# List secrets
vault kv list secret/myapp/
```

## Dynamic Database Secrets

Vault generates short-lived database credentials on demand:

```bash
# Configure database engine
vault write database/config/mydb \
    plugin_name=postgresql-database-plugin \
    connection_url="postgresql://{{username}}:{{password}}@db:5432/mydb" \
    allowed_roles="readonly" \
    username="vault_admin" \
    password="vault_pass"

# Create role
vault write database/roles/readonly \
    db_name=mydb \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"

# Get dynamic credentials
vault read database/creds/readonly
```

## Environment Variable Injection

Never hardcode secrets in application code. Use environment variables or secret files:

```python
import os

DB_PASSWORD = os.environ["DB_PASSWORD"]  # fail fast if missing
API_KEY = os.environ.get("API_KEY", "")  # optional with default
```

For Kubernetes, use external-secrets-operator to sync Vault secrets to K8s Secrets:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-creds
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: db-creds
  data:
    - secretKey: password
      remoteRef:
        key: secret/myapp/db
        property: password
```

## Git Secret Scanning

Prevent secrets from being committed:

```bash
# Use gitleaks
gitleaks detect --source . --verbose

# Pre-commit hook
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```
