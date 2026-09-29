from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://postgres:postgres@localhost:5432/rag"
)

def init_db():
    with engine.begin() as db:
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        db.execute(
            text("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    doc_type TEXT,
                    doc_date TIMESTAMPTZ,
                    embedding vector(3072)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant);

                CREATE TABLE IF NOT EXISTS ai_audit_log (
                    id SERIAL PRIMARY KEY,
                    tenant TEXT,
                    query TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost DOUBLE PRECISION,
                    chunks_used INTEGER,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        )

# Initialize database schema if not already present
try:
    init_db()
except Exception as e:
    print(f"Warning: could not auto-initialize database: {e}")



def insert_chunk(tenant, source, title, content, embedding): 
    with engine.begin() as db:
        db.execute(
            text("""
                INSERT INTO chunks
                (tenant, source, title, content, embedding)
                VALUES
                (:tenant, :source, :title, :content, :embedding)
            """),
            {
                "tenant": tenant,
                "source": source,
                "title": title,
                "content": content.replace("\x00", "") if isinstance(content, str) else content,
                "embedding": str(embedding)
            }
        )

def search_chunks(
    tenant,
    embedding,
    limit=20,
    doc_type=None
):

    query = """
        SELECT
            content,
            source,
            title,
            doc_type,
            doc_date,
            1 - (
                embedding <=> CAST(:embedding AS vector)
            ) AS score
        FROM chunks
        WHERE tenant = :tenant
    """

    params = {
        "tenant": tenant,
        "embedding": str(embedding),
        "limit": limit
    }

    if doc_type:
        query += " AND doc_type = :doc_type"
        params["doc_type"] = doc_type

    query += """
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """

    with engine.begin() as db:
        return db.execute(
            text(query),
            params
        ).fetchall()

def insert_audit_log(
    tenant,
    query,
    model,
    input_tokens,
    output_tokens,
    cost,
    chunks_used
):
    try:
        with engine.begin() as db:
            db.execute(text("""
                INSERT INTO ai_audit_log
                (
                    tenant,
                    query,
                    model,
                    input_tokens,
                    output_tokens,
                    cost,
                    chunks_used
                )
                VALUES
                (
                    :tenant,
                    :query,
                    :model,
                    :input_tokens,
                    :output_tokens,
                    :cost,
                    :chunks_used
                )
            """), {
                "tenant": tenant,
                "query": query,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "chunks_used": chunks_used
            })
    except Exception as e:
        print(f"Warning: Failed to insert audit log: {e}")



def get_db_stats(tenant="default"):
    try:
        with engine.begin() as db:
            chunks_count = db.execute(
                text("SELECT count(*) FROM chunks WHERE tenant = :tenant"),
                {"tenant": tenant}
            ).scalar() or 0

            docs_count = db.execute(
                text("SELECT count(DISTINCT source) FROM chunks WHERE tenant = :tenant"),
                {"tenant": tenant}
            ).scalar() or 0

            return {
                "status": "online",
                "chunks_count": chunks_count,
                "documents_count": docs_count
            }
    except Exception as e:
        return {
            "status": "offline",
            "error": str(e),
            "chunks_count": 0,
            "documents_count": 0
        }


def get_document_stats(tenant="default"):
    try:
        with engine.begin() as db:
            rows = db.execute(
                text("""
                    SELECT source, count(*) as count, MAX(doc_date) as latest_date
                    FROM chunks
                    WHERE tenant = :tenant
                    GROUP BY source
                    ORDER BY count DESC
                """),
                {"tenant": tenant}
            ).fetchall()
            return [
                {
                    "source": r[0],
                    "chunks": r[1],
                    "latest_date": str(r[2]) if r[2] else None
                }
                for r in rows
            ]
    except Exception as e:
        print(f"Error fetching document stats: {e}")
        return []


def delete_chunks_by_source(source, tenant="default"):
    with engine.begin() as db:
        res = db.execute(
            text("DELETE FROM chunks WHERE source = :source AND tenant = :tenant"),
            {"source": source, "tenant": tenant}
        )
        return res.rowcount


def truncate_chunks(tenant=None):
    with engine.begin() as db:
        if tenant:
            res = db.execute(
                text("DELETE FROM chunks WHERE tenant = :tenant"),
                {"tenant": tenant}
            )
            return res.rowcount
        else:
            db.execute(text("TRUNCATE TABLE chunks RESTART IDENTITY;"))
            return True


def get_recent_audit_logs(tenant="default", limit=10):
    try:
        with engine.begin() as db:
            rows = db.execute(
                text("""
                    SELECT id, query, model, input_tokens, output_tokens, cost, chunks_used, created_at
                    FROM ai_audit_log
                    WHERE tenant = :tenant OR tenant IS NULL
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"tenant": tenant, "limit": limit}
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "query": r[1],
                    "model": r[2],
                    "input_tokens": r[3],
                    "output_tokens": r[4],
                    "cost": r[5],
                    "chunks_used": r[6],
                    "created_at": str(r[7]) if r[7] else None
                }
                for r in rows
            ]
    except Exception as e:
        print(f"Error reading audit log: {e}")
        return []