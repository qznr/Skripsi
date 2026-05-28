import asyncio
import json
import os
import re
import click
import pandas as pd
from flask import Blueprint
from db import get_db
from psycopg2.extras import execute_values, RealDictCursor
from pgvector.psycopg2 import register_vector
from tqdm import tqdm
from services import find_experts, save_evaluation_results
from vectorizer import get_embeddings, get_single_embedding
import itertools
from llm_judge import evaluate_with_gemini, MODEL_NAME
import time

bp = Blueprint('commands', __name__)

@bp.cli.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables from schema.sql."""
    db_conn = get_db()
    cursor = db_conn.cursor()
    
    with open('schema.sql', 'r') as f:
        cursor.execute(f.read())
    
    db_conn.commit()
    cursor.close()
    click.echo('Initialized the database.')

@bp.cli.command('import-data')
def import_data():
    """Parses CSV files and imports them into the database using bulk operations."""
    data_dir = '/app/data'
    print(f"Searching for CSV files in {data_dir}...")

    db_conn = get_db()
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for filename in tqdm(csv_files, desc="Total Files Progress", unit="file"):
        file_path = os.path.join(data_dir, filename)
        tqdm.write(f"Processing file: {filename}...")

        df = pd.read_csv(file_path, keep_default_na=False).where(pd.notna, None)
        
        # STAGE 1: EXTRACT ALL UNIQUE ENTITIES IN-MEMORY
        all_authors = set()
        for author_string in df['Author full names'].dropna():
            for entry in author_string.split(';'):
                match = re.match(r'(.+?)\s*\((\d+)\)', entry.strip())
                if match:
                    all_authors.add((match.group(2).strip(), match.group(1).strip()))

        with db_conn.cursor() as cursor:
            # STAGE 2: BULK UPSERT & BUILD ID CACHES
            tqdm.write(f"  Syncing {len(all_authors)} unique authors...")
            execute_values(cursor, "INSERT INTO authors (scopus_id, full_name) VALUES %s ON CONFLICT (scopus_id) DO NOTHING;", list(all_authors))
            cursor.execute("SELECT scopus_id, author_id FROM authors;")
            author_id_cache = dict(cursor.fetchall())

            # STAGE 3: IDENTIFY & BULK INSERT NEW ARTICLES
            cursor.execute("SELECT eid FROM articles;")
            existing_eids = {row[0] for row in cursor.fetchall()}
            
            df_new = df[~df['EID'].isin(existing_eids)]
            if df_new.empty:
                tqdm.write(f"  No new articles to insert from {filename}.")
                continue

            tqdm.write(f"  Found {len(df_new)} new articles to process.")
            article_values = []
            
            for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc=f"    L Preparing Articles", unit="row", leave=False):
                open_access = bool(row.get('Open Access'))
                abstract = re.sub(r'\s*©.*$', '', row.get('Abstract'), flags=re.DOTALL).strip()
                article_values.append((
                    row['EID'], row.get('Title'), row.get('Year'), row.get('Source title'),
                    row.get('Cited by'), row.get('DOI'), row.get('Link'), abstract,
                    row.get('Document Type'), row.get('Source'), open_access,
                    row.get('ISSN'), row.get('ISBN'), row.get('CODEN')
                ))

            insert_query = """
                INSERT INTO articles (
                    eid, title, year, source_title, cited_by, doi, link, 
                    abstract, document_type, source, open_access, issn, isbn, coden
                ) VALUES %s RETURNING eid, article_id;
            """
            new_article_ids = execute_values(cursor, insert_query, article_values, fetch=True)
            eid_to_article_id = dict(new_article_ids)

            # STAGE 4: PREPARE LINKING TABLE DATA IN-MEMORY
            articles_authors_values = []

            for _, row in tqdm(df_new.iterrows(), total=len(df_new), desc=f"    L Preparing Links   ", unit="row", leave=False):
                article_id = eid_to_article_id.get(row['EID'])
                if not article_id:
                    continue

                # --- Author-Article Links ---
                if row.get("Author full names"):
                    authors_data = []
                    author_entries = str(row["Author full names"]).split(';')
                    for entry in author_entries:
                        match = re.match(r'(.+?)\s*\((\d+)\)', entry.strip())
                        if match:
                            authors_data.append({'full_name': match.group(1).strip(), 'scopus_id': match.group(2).strip()})

                    for i, author in enumerate(authors_data):
                        author_id = author_id_cache.get(author['scopus_id'])
                        if author_id:
                            articles_authors_values.append((article_id, author_id, i + 1))

            # STAGE 5: BULK INSERT LINKING DATA
            if articles_authors_values:
                tqdm.write(f"  Inserting {len(articles_authors_values)} author-article links...")
                execute_values(cursor, "INSERT INTO articles_authors (article_id, author_id, author_order) VALUES %s;", articles_authors_values)

            db_conn.commit()
            tqdm.write(f"Successfully committed data from {filename}")

    print("Data import process finished.")

def generate_batches(cursor, batch_size):
    """A generator function to yield batches of (ids, documents) from the database cursor."""
    while True:
        records = cursor.fetchmany(batch_size)
        if not records:
            break
        
        article_ids = [r[0] for r in records]
        documents = [
            f"search_document: {title or ''}\n{abstract or ''}".strip()
            for _, title, abstract in records
        ]
        yield article_ids, documents

async def _vectorize_documents(limit, batch_size, concurrency):
    """Generates and stores embeddings for articles in the database."""

    db_conn = get_db()
    register_vector(db_conn)

    with db_conn.cursor() as cursor:
        count_query = "SELECT COUNT(*) FROM articles WHERE embedding IS NULL;"
        cursor.execute(count_query)
        total_to_process = cursor.fetchone()[0]

    if total_to_process == 0:
        click.echo("No new documents to vectorize.")
        return

    process_limit = total_to_process if limit is None else min(limit, total_to_process)
    click.echo(f"Found {total_to_process} un-vectorized articles. Processing {process_limit}.")

    update_data_buffer = []
    db_update_threshold_batches = 32 
    
    with db_conn.cursor('vectorize_cursor', withhold=True) as cursor:
        query = "SELECT article_id, title, abstract FROM articles WHERE embedding IS NULL"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)

        batches_iterator = generate_batches(cursor, batch_size)

        with tqdm(total=process_limit, desc="Vectorizing Documents", unit="doc") as pbar:
            async for article_ids, embeddings in get_embeddings(batches_iterator, concurrency):
                
                update_data_buffer.extend(zip(article_ids, embeddings))
                pbar.update(len(article_ids))

                if len(update_data_buffer) >= db_update_threshold_batches * batch_size:
                    with db_conn.cursor() as update_cursor:
                        update_query = """
                            UPDATE articles SET embedding = data.embedding
                            FROM (VALUES %s) AS data (article_id, embedding)
                            WHERE articles.article_id = data.article_id;
                        """
                        execute_values(update_cursor, update_query, update_data_buffer, template='(%s, %s::vector)')
                    db_conn.commit()
                    update_data_buffer.clear()

    if update_data_buffer:
        with db_conn.cursor() as update_cursor:
            update_query = """
                UPDATE articles SET embedding = data.embedding
                FROM (VALUES %s) AS data (article_id, embedding)
                WHERE articles.article_id = data.article_id;
            """
            execute_values(update_cursor, update_query, update_data_buffer, template='(%s, %s::vector)')
        db_conn.commit()

    click.echo("Vectorization complete.")

@bp.cli.command('vectorize')
@click.option('--limit', '-n', default=None, type=int, help='Number of documents to vectorize.')
@click.option('--batch-size', default=8, type=int, help='Batch size for the embedding model.')
@click.option('--concurrency', default=128, type=int, help='Number of concurrent requests.')
def vectorize_documents(limit, batch_size, concurrency):
    """Generates and stores embeddings for articles in the database."""
    asyncio.run(_vectorize_documents(limit, batch_size, concurrency))
    click.echo("Vectorization complete.")

@bp.cli.command('add-db-functions')
def add_db_functions_command():
    """Adds or updates custom SQL functions to the database without dropping tables."""
    db_conn = get_db()
    
    sub_vector_func = """
    CREATE OR REPLACE FUNCTION sub_vector(vec VECTOR, dims INT)
    RETURNS VECTOR AS $$
    DECLARE
        vec_as_array FLOAT4[];
    BEGIN
        vec_as_array := vec::FLOAT4[];

        IF dims > array_length(vec_as_array, 1) THEN
            RAISE EXCEPTION 'dimensions must be less than or equal to the vector size of %', array_length(vec_as_array, 1);
        END IF;

        RETURN (
            WITH unnormed(elem) AS (
                SELECT x
                FROM unnest(vec_as_array) WITH ORDINALITY v(x, ix)
                WHERE ix <= dims
            ),
            norm(factor) AS (
                SELECT SQRT(SUM(POW(elem, 2)))
                FROM unnormed
            )
            SELECT ARRAY_AGG(u.elem / r.factor)
            FROM norm r, unnormed u
            WHERE r.factor != 0
        );
    END;
    $$ LANGUAGE plpgsql IMMUTABLE;
    """
        
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(sub_vector_func)
        db_conn.commit()
        click.echo('Successfully added/updated database functions.')
    except Exception as e:
        click.echo(f"An error occurred: {e}")
        db_conn.rollback()

@bp.cli.command('create-hnsw-index')
@click.option('--m', default=32, type=int, help='Number of bi-directional links created for each new element (HNSW parameter M).')
@click.option('--ef-construction', default=400, type=int, help='Construction effort parameter for HNSW.')
def create_hnsw_index(m, ef_construction):
    """Creates HNSW indexes on sub-vectors of different dimensions (768, 512, 256, 128, 64)."""
    db_conn = get_db()

    dims_list = [768, 512, 256, 128, 64]

    try:
        with db_conn.cursor() as cursor:
            for dims in dims_list:
                index_name = f"idx_articles_embedding_hnsw_{dims}"
                create_index_sql = f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON articles
                USING hnsw ((sub_vector(embedding, {dims})::vector({dims})) vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction});
                """
                cursor.execute(create_index_sql)
                click.echo(f"Created/exists HNSW index for {dims} dimensions (M={m}, efConstruction={ef_construction}).")

        db_conn.commit()
        click.echo("All HNSW indexes created successfully.")
    except Exception as e:
        click.echo(f"Failed to create HNSW indexes: {e}")
        db_conn.rollback()

@bp.cli.command('add-models')
def add_models_command():
    """Adds the standard set of evaluation models to the database."""
    db_conn = get_db()
    models_to_add = [
        ("64_dim", "Model using 64-dimensional vectors."),
        ("128_dim", "Model using 128-dimensional vectors."),
        ("256_dim", "Model using 256-dimensional vectors."),
        ("512_dim", "Model using 512-dimensional vectors."),
        ("768_dim", "Model using 768-dimensional vectors."),
    ]
    
    insert_query = "INSERT INTO models (model_name, description) VALUES (%s, %s) ON CONFLICT (model_name) DO NOTHING;"
    
    try:
        with db_conn.cursor() as cursor:
            for name, desc in models_to_add:
                cursor.execute(insert_query, (name, desc))
                if cursor.rowcount > 0:
                    click.echo(f"Added model: {name}")
                else:
                    click.echo(f"Model '{name}' already exists.")
        db_conn.commit()
        click.echo("Finished adding models.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")
        db_conn.rollback()

@bp.cli.command('upgrade-schema')
def upgrade_schema_command():
    """Adds new tables and refactors existing tables non-destructively."""
    db_conn = get_db()
    
    # 1. Original Table Upgrades
    new_tables_sql = """
    CREATE TABLE IF NOT EXISTS articles_expert_results (
        article_expert_result_id SERIAL PRIMARY KEY,
        expert_result_id INT REFERENCES expert_results(result_id),
        article_id INT REFERENCES articles(article_id),
        similarity_score FLOAT NOT NULL,
        UNIQUE (expert_result_id, article_id)
    );
    """
    
    # 2. Latency Refactoring & Preference Flag
    refactor_eval_sql = """
    DO $$
    BEGIN
        ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS latency_a JSONB;
        ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS latency_b JSONB;
        
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='evaluation_results' AND column_name='latency_a' AND data_type='double precision') THEN
             ALTER TABLE evaluation_results DROP COLUMN latency_a;
        END IF;
        IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='evaluation_results' AND column_name='latency_b' AND data_type='double precision') THEN
             ALTER TABLE evaluation_results DROP COLUMN latency_b;
        END IF;

        ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS preference_submitted BOOLEAN DEFAULT FALSE;
    END
    $$;
    """

    # 3. LLM Benchmarking Columns
    llm_columns_sql = """
    ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS evaluator_type VARCHAR(50) DEFAULT 'human';
    ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS evaluator_model VARCHAR(100);
    ALTER TABLE evaluation_results ADD COLUMN IF NOT EXISTS reasoning TEXT;

    ALTER TABLE expert_results ADD COLUMN IF NOT EXISTS model_id INT REFERENCES models(model_id);
    """
    
    try:
        with db_conn.cursor() as cursor:
            click.echo('Running schema upgrades...')
            cursor.execute(new_tables_sql)
            cursor.execute(refactor_eval_sql)
            cursor.execute(llm_columns_sql)
            
        db_conn.commit()
    except Exception as e:
        click.echo(f"An error occurred during schema upgrade: {e}")
        db_conn.rollback()

async def _run_benchmark():
    conn = get_db()
    
    with conn.cursor() as cursor:
        cursor.execute("SELECT q.query_text, model_a_id, model_b_id FROM evaluation_results e JOIN queries q ON e.query_id = q.query_id WHERE evaluator_type = 'llm';")
        existing_matches = {(row[0], row[1], row[2]) for row in cursor.fetchall()}

    with open('/app/data/benchmark_queries.json', 'r') as f:
        queries = json.load(f)
        
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT model_id, model_name FROM models ORDER BY model_id;")
        all_models = cursor.fetchall()
        
    model_pairs = list(itertools.combinations(all_models, 2))
    
    for q_idx, query_text in enumerate(queries):
        # Prefetch Query Embedding
        prefixed_query = f"search_query: {query_text}"
        query_embedding = await get_single_embedding(prefixed_query)
        
        print(f"\n=======================================================")
        print(f"Testing [{q_idx+1}/{len(queries)}] Query: '{query_text}'")
        print(f"=======================================================")

        # ---------------------------------------------------------
        # STEP 1: ISOLATED TIMING RUN (One per dimension)
        # ---------------------------------------------------------
        # We run the DB search exactly ONCE per model for this query
        # to ensure fair, real-world latency metrics without loop-cache bias.
        
        query_results_cache = {}
        query_metrics_cache = {}
        
        for model in all_models:
            dims = int(model['model_name'].split('_')[0])
            res, met = await find_experts(query_text, dims, 100)
            
            query_results_cache[model['model_id']] = res
            query_metrics_cache[model['model_id']] = met
            
        # ---------------------------------------------------------
        # STEP 2: GEMINI JUDGING (Using the isolated data)
        # ---------------------------------------------------------
        for pair_idx, (model_a, model_b) in enumerate(model_pairs):
            
            # --- SKIP CHECK ---
            if (query_text, model_a['model_id'], model_b['model_id']) in existing_matches:
                print(f"  [Skipping] {model_a['model_name']} vs {model_b['model_name']} - Already judged.")
                continue

            print(f"\n  [Pairing] {model_a['model_name']} vs {model_b['model_name']}")
            
            # Retrieve the cleanly timed data from our Python dictionary
            results_a = query_results_cache[model_a['model_id']]
            metrics_a = query_metrics_cache[model_a['model_id']]
            
            results_b = query_results_cache[model_b['model_id']]
            metrics_b = query_metrics_cache[model_b['model_id']]
            
            results_identical = (results_a == results_b)
            
            # Call Gemini
            print("    -> Waiting for Gemini Judge...")
            gemini_response = evaluate_with_gemini(query_text, results_a, results_b)
            
            choice = gemini_response.get("choice")
            reasoning = gemini_response.get("reasoning")
            
            winner_id = None
            preference_submitted = False
            
            if choice:
                preference_submitted = True
                print(f"    -> Gemini Voted: {choice.upper()} | Reason: {reasoning[:60]}...")
                if choice == 'a': winner_id = model_a['model_id']
                elif choice == 'b': winner_id = model_b['model_id']
            else:
                print(f"    -> Gemini FAILED. Skipping recording for this match.")
                time.sleep(30) 
                continue 

            # --- SAVE DATA USING THE UNIFIED HELPER ---
            eval_id = save_evaluation_results(
                conn, query_embedding, query_text, model_a, model_b,
                results_a, results_b, metrics_a, metrics_b,
                evaluator_type='llm', evaluator_model=MODEL_NAME, reasoning=reasoning
            )
            
            # If Gemini picked a winner, update the database record with the winner_id
            if winner_id:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "UPDATE evaluation_results SET winner_model_id = %s WHERE evaluation_id = %s;",
                        (winner_id, eval_id)
                    )
                conn.commit()

    print("\nBenchmark Process Finished.")

@bp.cli.command('run-benchmark')
def run_benchmark_command():
    """Automates the LLM-as-a-Judge evaluation across all dimension pairs."""
    asyncio.run(_run_benchmark())