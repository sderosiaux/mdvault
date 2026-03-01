# CI/CD Pipeline Design

## GitHub Actions

A comprehensive workflow for a Python project:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v1

  deploy:
    needs: [test, lint]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp:${{ github.sha }} .
      - run: docker push registry.example.com/myapp:${{ github.sha }}
```

## Pipeline Best Practices

- **Fast feedback**: Run linting and unit tests first, integration tests later
- **Caching**: Cache dependency installations between runs
- **Matrix builds**: Test against multiple Python/Node versions
- **Branch protection**: Require passing CI before merge
- **Secrets management**: Use GitHub Secrets or Vault for credentials

## Deployment Strategies

**Blue-Green**: Two identical environments. Switch traffic atomically.

**Canary**: Route a small percentage of traffic to the new version. Gradually increase if metrics are healthy.

**Rolling**: Update instances one at a time. No downtime but mixed versions during deployment.

## ArgoCD for GitOps

Declarative continuous delivery for Kubernetes:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/k8s-manifests
    targetRevision: main
    path: apps/myapp
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```
