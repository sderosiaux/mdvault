# Docker Best Practices

## Dockerfile Optimization

Write efficient Dockerfiles by leveraging layer caching and multi-stage builds:

```dockerfile
# Stage 1: Build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --production=false
COPY . .
RUN npm run build

# Stage 2: Production
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

Order instructions from least to most frequently changing. `COPY package*.json` before `COPY .` ensures npm install is cached unless dependencies change.

## Docker Compose Networking

Docker Compose creates a default bridge network. Services communicate using their service names as hostnames:

```yaml
services:
  web:
    build: .
    ports:
      - "8080:3000"
    depends_on:
      db:
        condition: service_healthy
    networks:
      - frontend
      - backend

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
    networks:
      - backend

networks:
  frontend:
  backend:
    internal: true

volumes:
  pgdata:
```

The `internal: true` flag on the backend network prevents external access, isolating the database.

## Volume Management

Volumes persist data beyond container lifecycle:

```bash
# Create named volume
docker volume create pgdata

# Inspect volume
docker volume inspect pgdata

# Backup volume
docker run --rm -v pgdata:/data -v $(pwd):/backup alpine tar czf /backup/pgdata.tar.gz -C /data .

# Restore volume
docker run --rm -v pgdata:/data -v $(pwd):/backup alpine tar xzf /backup/pgdata.tar.gz -C /data
```

Use named volumes over bind mounts for production data. Bind mounts are fine for development hot-reloading.

## Resource Limits

Set memory and CPU limits to prevent runaway containers:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
```
