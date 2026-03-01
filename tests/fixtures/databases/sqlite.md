# SQLite Tips and Tricks

## WAL Mode

Write-Ahead Logging improves concurrent read performance:

```sql
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;  -- 64MB
PRAGMA foreign_keys = ON;
```

WAL mode allows concurrent readers while a single writer is active. This is the recommended mode for most applications.

## JSON Support

SQLite has built-in JSON functions:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    data TEXT NOT NULL  -- JSON string
);

-- Extract JSON field
SELECT json_extract(data, '$.user.name') FROM events;

-- Query JSON arrays
SELECT * FROM events, json_each(json_extract(data, '$.tags'))
WHERE json_each.value = 'important';
```

## Full-Text Search with FTS5

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(title, content);
INSERT INTO docs_fts VALUES ('SQLite Guide', 'SQLite is a lightweight database...');

-- Search
SELECT * FROM docs_fts WHERE docs_fts MATCH 'lightweight database';

-- BM25 ranking
SELECT *, rank FROM docs_fts WHERE docs_fts MATCH 'database' ORDER BY rank;
```
