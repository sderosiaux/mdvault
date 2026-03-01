# PostgreSQL Administration

## Performance Tuning

Key postgresql.conf parameters for a server with 32GB RAM:

```
shared_buffers = 8GB              # 25% of RAM
effective_cache_size = 24GB       # 75% of RAM
work_mem = 64MB                   # per-operation sort memory
maintenance_work_mem = 2GB        # for VACUUM, CREATE INDEX
wal_buffers = 64MB
max_connections = 200
random_page_cost = 1.1            # for SSD storage
effective_io_concurrency = 200    # for SSD storage
```

Use `pg_stat_statements` to identify slow queries:

```sql
CREATE EXTENSION pg_stat_statements;
SELECT query, calls, mean_exec_time, total_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

## Backup and Recovery

Use `pg_dump` for logical backups and `pg_basebackup` for physical backups:

```bash
# Logical backup (single database)
pg_dump -h localhost -U postgres -Fc mydb > mydb.dump

# Restore logical backup
pg_restore -h localhost -U postgres -d mydb mydb.dump

# Physical backup (entire cluster)
pg_basebackup -h primary -U replicator -D /backup/base -Fp -Xs -P
```

For point-in-time recovery (PITR), enable WAL archiving:

```
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

Then restore to a specific timestamp:

```
restore_command = 'cp /archive/%f %p'
recovery_target_time = '2024-01-15 14:30:00'
```

## Streaming Replication

Set up a read replica for scaling reads and high availability:

On the primary server:

```
wal_level = replica
max_wal_senders = 5
wal_keep_size = 1GB
```

Create replication user:

```sql
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'replpass';
```

On the replica:

```bash
pg_basebackup -h primary-host -U replicator -D /var/lib/postgresql/16/main -Fp -Xs -P -R
```

The `-R` flag creates `standby.signal` and configures `primary_conninfo` automatically.

## Indexing Strategies

Choose the right index type for your query patterns:

```sql
-- B-tree (default, equality and range queries)
CREATE INDEX idx_users_email ON users(email);

-- GIN (full-text search, arrays, JSONB)
CREATE INDEX idx_docs_content ON documents USING gin(to_tsvector('english', content));

-- BRIN (large tables with natural ordering)
CREATE INDEX idx_events_created ON events USING brin(created_at);

-- Partial index (subset of rows)
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

## Connection Pooling

Use PgBouncer for connection pooling:

```ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

Transaction pooling mode is recommended for most applications. Session pooling is needed for prepared statements.
