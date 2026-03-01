# Redis Caching and Data Patterns

## Caching Strategies

Redis is commonly used as an application cache layer. The two primary patterns:

**Cache-aside (lazy loading)**: Application checks cache first, loads from DB on miss:

```python
def get_user(user_id):
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**Write-through**: Application writes to cache and DB simultaneously:

```python
def update_user(user_id, data):
    db.execute("UPDATE users SET ... WHERE id = %s", user_id)
    redis.setex(f"user:{user_id}", 3600, json.dumps(data))
```

Cache-aside is simpler and avoids caching unused data. Write-through ensures cache freshness at the cost of write latency.

## Persistence Options

Redis offers two persistence mechanisms:

- **RDB snapshots**: Point-in-time snapshots at intervals. Fast restart but potential data loss.
- **AOF (Append Only File)**: Logs every write. More durable but larger files and slower restart.

Recommended production config:

```
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec
```

## Data Structures

Redis supports rich data structures beyond simple key-value:

```bash
# Sorted set for leaderboards
ZADD leaderboard 1500 "player:alice"
ZADD leaderboard 2100 "player:bob"
ZREVRANGE leaderboard 0 9 WITHSCORES

# Hash for objects
HSET user:1001 name "Alice" email "alice@example.com" role "admin"
HGETALL user:1001

# List as a queue
LPUSH jobs '{"type":"email","to":"bob@example.com"}'
BRPOP jobs 30

# HyperLogLog for cardinality estimation
PFADD daily_visitors "user:123" "user:456"
PFCOUNT daily_visitors
```

## Eviction Policies

When memory limit is reached, Redis evicts keys based on the configured policy:

- `allkeys-lru`: Evict least recently used keys (recommended for caches)
- `volatile-lru`: Evict LRU keys with TTL set
- `allkeys-lfu`: Evict least frequently used keys
- `noeviction`: Return errors on write (for data stores)

```
maxmemory 4gb
maxmemory-policy allkeys-lru
```
