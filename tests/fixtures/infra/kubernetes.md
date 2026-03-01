# Kubernetes Operations Guide

## Pod Configuration

A pod is the smallest deployable unit in Kubernetes. Define resource requests and limits for every container:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-app
  labels:
    app: web
spec:
  containers:
    - name: web
      image: myapp:1.2.0
      ports:
        - containerPort: 8080
      resources:
        requests:
          memory: "128Mi"
          cpu: "250m"
        limits:
          memory: "256Mi"
          cpu: "500m"
      livenessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 10
        periodSeconds: 15
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 10
```

Always set both liveness and readiness probes. Liveness probes restart unhealthy containers. Readiness probes remove pods from service endpoints until they are ready to accept traffic.

## Services and Networking

Services provide stable network endpoints for pods:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8080
```

Service types: `ClusterIP` (internal only), `NodePort` (expose on each node), `LoadBalancer` (cloud provider LB). Use `ClusterIP` for inter-service communication and `LoadBalancer` or `Ingress` for external traffic.

## Ingress Controller

Ingress routes external HTTP/HTTPS traffic to services:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - app.example.com
      secretName: tls-secret
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80
```

## Horizontal Pod Autoscaler

Scale pods based on CPU or custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## ConfigMaps and Secrets

Externalize configuration from container images:

```bash
# Create ConfigMap from file
kubectl create configmap app-config --from-file=config.yaml

# Create Secret
kubectl create secret generic db-creds \
    --from-literal=username=admin \
    --from-literal=password=s3cur3
```
