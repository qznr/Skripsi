DROP TABLE IF EXISTS expert_results, articles_expert_results, evaluation_results, models, queries, articles_authors, authors, articles CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE articles (
    article_id SERIAL PRIMARY KEY,
    eid TEXT UNIQUE,
    title TEXT,
    year INT,
    source_title TEXT,
    cited_by INT,
    doi TEXT,
    link TEXT,
    abstract TEXT,
    document_type TEXT,
    open_access BOOLEAN,
    issn TEXT,
    isbn TEXT,
    coden TEXT,
    source TEXT,
    embedding VECTOR(768)
);

CREATE TABLE authors (
    author_id SERIAL PRIMARY KEY,
    full_name TEXT,
    scopus_id TEXT UNIQUE
);

CREATE TABLE articles_authors (
    article_id INT REFERENCES articles(article_id),
    author_id INT REFERENCES authors(author_id),
    author_order INT,
    PRIMARY KEY(article_id, author_id)
);

CREATE TABLE queries (
    query_id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE models (
    model_id SERIAL PRIMARY KEY,
    model_name TEXT UNIQUE NOT NULL,
    current_elo FLOAT DEFAULT 1000.0,
    model_version TEXT,
    description TEXT
);

CREATE TABLE evaluation_results (
    evaluation_id SERIAL PRIMARY KEY,
    query_id INT REFERENCES queries(query_id),
    model_a_id INT REFERENCES models(model_id),
    model_b_id INT REFERENCES models(model_id),
    winner_model_id INT REFERENCES models(model_id),
    latency_a JSONB,
    latency_b JSONB,
    results_identical BOOLEAN DEFAULT FALSE,
    preference_submitted BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE expert_results (
    result_id SERIAL PRIMARY KEY,
    evaluation_id INT REFERENCES evaluation_results(evaluation_id),
    author_id INT REFERENCES authors(author_id),
    rank_position INT NOT NULL
);

CREATE TABLE articles_expert_results (
    article_expert_result_id SERIAL PRIMARY KEY,
    expert_result_id INT REFERENCES expert_results(result_id),
    article_id INT REFERENCES articles(article_id),
    similarity_score FLOAT NOT NULL,
    UNIQUE (expert_result_id, article_id)
);