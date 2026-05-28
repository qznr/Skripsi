from db import get_db
from vectorizer import get_single_embedding
from psycopg2.extras import RealDictCursor, execute_values
import asyncio
import random
from collections import defaultdict
import time
import json

async def find_experts(
    query_text: str,
    initial_dims: int,
    shortlist_size: int = 25
):
    """
    Orchestrates a two-stage process of finding and ranking experts.

    Args:
        query_text (str): The user's search query.
        initial_dims (int): The vector dimension for the fast initial retrieval.
        shortlist_size (int): The number of candidates for the second stage.
    """
    conn = get_db()
    latency_metrics = {}

    # Check if reranking is needed
    perform_reranking = (initial_dims != 768)

    # === STEP 1: Get Query Embedding ===
    start_step1 = time.perf_counter()
    prefixed_query = f"search_query: {query_text}"
    query_embedding = await get_single_embedding(prefixed_query)
    latency_metrics['step1_embedding'] = time.perf_counter() - start_step1

    # === STEP 2: Initial Candidate Retrieval (Stage 1) ===
    # Fast retrieval using a lower-dimension index to get a shortlist of IDs.
    start_step2 = time.perf_counter()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        sql_initial_search = f"""
            SELECT a.article_id, a.title, a.abstract,
                   (sub_vector(a.embedding, %(initial_dims)s)::vector(%(initial_dims)s) <=> sub_vector(%(query_embedding)s::vector, %(initial_dims)s)::vector(%(initial_dims)s)) AS raw_distance
            FROM articles a
            WHERE a.embedding IS NOT NULL
            ORDER BY raw_distance ASC  -- Index-friendly ordering
            LIMIT %(shortlist_size)s;
        """
        cursor.execute(sql_initial_search, {
            'query_embedding': query_embedding,
            'initial_dims': initial_dims,
            'shortlist_size': shortlist_size
        })
        initial_candidates = cursor.fetchall()

    latency_metrics['step2_shortlisting'] = time.perf_counter() - start_step2
    
    if not initial_candidates:
        latency_metrics['total_search_time'] = sum(latency_metrics.values())
        return [], latency_metrics 

    # === STEP 3: Reranking (Stage 2) ===
    score_map = {}
    top_ranked_ids = []
    if perform_reranking:
        start_step3 = time.perf_counter()
        # Rerank the larger shortlist using precise, full-dimension vector similarity.
        candidate_ids = [doc['article_id'] for doc in initial_candidates]
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            sql_rerank = """
                SELECT
                    article_id,
                    1 - (embedding <=> %(query_embedding)s::vector(768)) AS similarity_score
                FROM articles
                WHERE article_id = ANY(%(candidate_ids)s)
                ORDER BY similarity_score DESC;
            """
            cursor.execute(sql_rerank, {
                'query_embedding': query_embedding,
                'candidate_ids': candidate_ids
            })
            reranked_docs = cursor.fetchall()
            top_ranked_ids = [doc['article_id'] for doc in reranked_docs]
            score_map = {doc['article_id']: doc['similarity_score'] for doc in reranked_docs}
        latency_metrics['step3_reranking'] = time.perf_counter() - start_step3
    else: 
        # === STEP 3: Reranking (Skipped for 768-dim model) ===
        latency_metrics['step3_reranking'] = 0.0
        
        # If reranking is skipped, the initial candidates are the final candidates.
        # We need to calculate the final similarity score (which is just the initial score)
        # and set the top_ranked_ids/score_map based on the initial retrieval.
        top_ranked_ids = [doc['article_id'] for doc in initial_candidates]
        score_map = {doc['article_id']: doc['raw_distance'] for doc in initial_candidates}

    # === STEP 4: Fetch Author and Article Data for Aggregation ===
    start_step4 = time.perf_counter()
    author_contributions = []
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        sql_fetch_details = """
            SELECT
                au.author_id, au.full_name, au.scopus_id,
                a.article_id, a.title, a.year, a.source_title, a.link, a.abstract,
                aa.author_order,
                a.cited_by
            FROM articles a
            JOIN articles_authors aa ON a.article_id = aa.article_id
            JOIN authors au ON aa.author_id = au.author_id
            WHERE a.article_id = ANY(%(article_ids)s);
        """
        cursor.execute(sql_fetch_details, {'article_ids': top_ranked_ids})
        author_contributions = cursor.fetchall()
    latency_metrics['step4_fetch_details'] = time.perf_counter() - start_step4

    # === STEP 5: Aggregate Scores in Python ===
    start_step5 = time.perf_counter()
    expert_scores = defaultdict(float)
    expert_articles = defaultdict(list)
    expert_info = {}

    for contrib in author_contributions:
        author_id = contrib['author_id']
        article_id = contrib['article_id']
        
        if author_id not in expert_info:
            expert_info[author_id] = {
                'author_id': author_id,
                'full_name': contrib['full_name'],
                'scopus_id': contrib['scopus_id']
            }
        
        normalized_score = score_map.get(article_id, 0)
        author_order_weight = 1.0
        order = contrib['author_order']
        if order == 2: author_order_weight = 0.8
        elif order == 3: author_order_weight = 0.6
        elif order > 3: author_order_weight = 0.4
        
        weighted_score = normalized_score * author_order_weight
        expert_scores[author_id] += weighted_score

        expert_articles[author_id].append({
            'article_id': article_id,
            'title': contrib['title'],
            'year': contrib['year'],
            'source_title': contrib['source_title'],
            'link': contrib['link'],
            'similarity_score': normalized_score,
            'abstract': contrib['abstract'],
            'cited_by': contrib['cited_by'] or 0,
        })
    latency_metrics['step5_score_aggregation'] = time.perf_counter() - start_step5

    # === STEP 6: Format Final Results ===
    start_step6 = time.perf_counter()
    final_results = []
    for author_id, total_score in expert_scores.items():
        sorted_articles = sorted(expert_articles[author_id], key=lambda x: x['similarity_score'], reverse=True)
        
        author_data = expert_info[author_id]
        author_data['expert_score'] = total_score
        author_data['articles'] = sorted_articles
        final_results.append(author_data)

    final_results.sort(key=lambda x: x['expert_score'], reverse=True)
    latency_metrics['step6_final_format'] = time.perf_counter() - start_step6

    latency_metrics['total_search_time'] = sum(latency_metrics[key] for key in latency_metrics if key.startswith('step'))

    return final_results[:5], latency_metrics

async def run_pairwise_search(query_text: str):
    """
    Orchestrates a pairwise search comparing two randomly selected retrieval models.
    Saves detailed, granular results and latency metrics to the database.
    """
    conn = get_db()
    
    # 0. Get Query Embedding (needed for saving to DB)
    prefixed_query = f"search_query: {query_text}"
    query_embedding = await get_single_embedding(prefixed_query)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT model_id, model_name FROM models;")
        all_models = cursor.fetchall()

    if len(all_models) < 2:
        raise ValueError("Not enough models configured for a pairwise test. At least 2 are required.")
    
    # Randomly select two models
    model_a_data, model_b_data = random.sample(all_models, 2)
    
    try:
        dims_a = int(model_a_data['model_name'].split('_')[0])
        dims_b = int(model_b_data['model_name'].split('_')[0])
    except (ValueError, IndexError):
        raise ValueError("Model name format is incorrect. Expected format like '256_dim'.")

    # Run tasks concurrently
    task_a = find_experts(
        query_text=query_text,
        initial_dims=dims_a,
        shortlist_size=100
    )
    task_b = find_experts(
        query_text=query_text,
        initial_dims=dims_b,
        shortlist_size=100
    )
    
    # Await both tasks. The results are tuples: (results, metrics)
    results_a_tuple, results_b_tuple = await asyncio.gather(task_a, task_b)
    results_a, metrics_a = results_a_tuple
    results_b, metrics_b = results_b_tuple

    # --- Save Results ---
    evaluation_id = _save_evaluation_results(
        conn, query_embedding, query_text, 
        model_a_data, model_b_data, 
        results_a, results_b, 
        metrics_a, metrics_b
    )

    return {
        "evaluation_id": evaluation_id,
        "model_a": {
            "model_id": model_a_data['model_id'],
            "model_name": model_a_data['model_name'],
            "experts": results_a
        },
        "model_b": {
            "model_id": model_b_data['model_id'],
            "model_name": model_b_data['model_name'],
            "experts": results_b
        }
    }

def _save_evaluation_results(conn, query_embedding, query_text, model_a_data, model_b_data, results_a, results_b, metrics_a, metrics_b):
    """
    Saves the entire evaluation structure using JSONB for latency metrics.
    """
    with conn.cursor() as cursor:
        # 1. Save Query
        cursor.execute("INSERT INTO queries (query_text, query_embedding) VALUES (%s, %s) RETURNING query_id;", 
                       (query_text, query_embedding))
        query_id = cursor.fetchone()[0]

        # 2. Save Evaluation (Winner is Null initially)
        evaluation_sql = """
            INSERT INTO evaluation_results 
            (query_id, model_a_id, model_b_id, 
             latency_a, latency_b, 
             results_identical) 
            VALUES (%s, %s, %s, %s, %s, %s) 
            RETURNING evaluation_id;
        """
        results_identical = (results_a == results_b)
        
        # Convert metrics dictionaries to JSON strings for JSONB insertion
        latency_a_json = json.dumps(metrics_a)
        latency_b_json = json.dumps(metrics_b)
        
        cursor.execute(evaluation_sql, (
            query_id, model_a_data['model_id'], model_b_data['model_id'], 
            latency_a_json, latency_b_json,
            results_identical
        ))
        evaluation_id = cursor.fetchone()[0]
        
        # Helper to process and save expert results for one model
        def save_model_results(results):
            expert_results_values = []
            article_expert_results_values = []
            
            # 3. Prepare Expert Results
            for rank, expert in enumerate(results):
                expert_results_values.append((evaluation_id, expert['author_id'], rank + 1))
            
            # Bulk insert expert results to get the generated result_id
            expert_insert_sql = """
                INSERT INTO expert_results (evaluation_id, author_id, rank_position) 
                VALUES %s 
                RETURNING result_id, author_id;
            """
            
            expert_id_map = execute_values(cursor, expert_insert_sql, expert_results_values, fetch=True)
            expert_result_id_by_author = {author_id: result_id for result_id, author_id in expert_id_map}
            
            # 4. Prepare Articles Expert Results
            for expert in results:
                expert_result_id = expert_result_id_by_author.get(expert['author_id'])
                if expert_result_id:
                    for article in expert['articles']:
                        article_expert_results_values.append((
                            expert_result_id,
                            article['article_id'],
                            article['similarity_score']
                        ))
            
            # 5. Bulk insert article details
            if article_expert_results_values:
                article_insert_sql = """
                    INSERT INTO articles_expert_results 
                    (expert_result_id, article_id, similarity_score) 
                    VALUES %s;
                """
                execute_values(cursor, article_insert_sql, article_expert_results_values)

        # Save results for both models
        save_model_results(results_a)
        save_model_results(results_b)

        conn.commit()
        return evaluation_id
    
def record_preference(evaluation_id: int, choice: str):
    """
    Updates the evaluation_results table with the user's choice.
    
    Args:
        evaluation_id: ID of the search session.
        choice: 'a', 'b', or 'draw'.
    """
    conn = get_db()
    
    # 1. Fetch model IDs for A and B to map the choice to winner_model_id
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT model_a_id, model_b_id FROM evaluation_results WHERE evaluation_id = %s;",
            (evaluation_id,)
        )
        result = cursor.fetchone()
        
        if result is None:
            raise ValueError(f"Evaluation ID {evaluation_id} not found.")

        model_a_id, model_b_id = result
        winner_id = None
        
        if choice == 'a':
            winner_id = model_a_id
        elif choice == 'b':
            winner_id = model_b_id
        # If choice is 'draw', winner_id remains None (representing no winner/draw)
        
        # 2. Update the evaluation record
        cursor.execute(
            """
            UPDATE evaluation_results
            SET winner_model_id = %s,
                preference_submitted = TRUE
            WHERE evaluation_id = %s;
            """,
            (winner_id, evaluation_id)
        )
        
        conn.commit()