-- Run as PostgreSQL superuser or db owner
-- This migration is idempotent (safe to re-run)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS clinical_nlp;

CREATE TABLE IF NOT EXISTS clinical_nlp.concepts (
    concept_id TEXT PRIMARY KEY,
    preferred_term TEXT NOT NULL,
    domain TEXT NOT NULL,
    source TEXT NOT NULL,
    hierarchy TEXT[],
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_concepts_domain ON clinical_nlp.concepts(domain);
CREATE INDEX IF NOT EXISTS idx_concepts_preferred_term ON clinical_nlp.concepts
    USING gin(preferred_term gin_trgm_ops);

CREATE TABLE IF NOT EXISTS clinical_nlp.synonyms (
    synonym_id SERIAL PRIMARY KEY,
    concept_id TEXT NOT NULL REFERENCES clinical_nlp.concepts(concept_id) ON DELETE CASCADE,
    synonym_text TEXT NOT NULL,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_synonyms_concept ON clinical_nlp.synonyms(concept_id);
CREATE INDEX IF NOT EXISTS idx_synonyms_text ON clinical_nlp.synonyms
    USING gin(synonym_text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS clinical_nlp.embeddings (
    concept_id TEXT PRIMARY KEY REFERENCES clinical_nlp.concepts(concept_id) ON DELETE CASCADE,
    embedding vector(384),
    model TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON clinical_nlp.embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS clinical_nlp.cache (
    term TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL REFERENCES clinical_nlp.concepts(concept_id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    cached_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ttl_days INTEGER DEFAULT 90
);
CREATE INDEX IF NOT EXISTS idx_cache_expiry ON clinical_nlp.cache(cached_at);

CREATE TABLE IF NOT EXISTS clinical_nlp.metadata (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION clinical_nlp.cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM clinical_nlp.cache
    WHERE cached_at + (ttl_days || ' days')::INTERVAL < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

GRANT USAGE ON SCHEMA clinical_nlp TO postgres;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA clinical_nlp TO postgres;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA clinical_nlp TO postgres;
