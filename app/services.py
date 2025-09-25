from db import get_db
from vectorizer import get_single_embedding, rerank_documents
from psycopg2.extras import RealDictCursor
import asyncio
import random
import time
from collections import defaultdict

async def find_experts(query_text: str, dims: int):
    """
    Orchestrates the process of finding and ranking experts using a reranking step.
    1.  Vectorizes the query.
    2.  Executes an initial vector search for the top 100 candidate articles.
    3.  Reranks the candidates using a dedicated reranker model.
    4.  Fetches author data for the top reranked articles.
    5.  Aggregates scores and formats the final results in Python.
    """
    conn = get_db()
    
    # === STEP 1: Get Query Embedding ===
    prefixed_query = f"search_query: {query_text}"
    query_embedding = await get_single_embedding(prefixed_query)

    # === STEP 2: Initial Candidate Retrieval (Top 100) ===
    initial_candidates = []
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        sql_initial_search = f"""
            SELECT
                a.article_id, a.title, a.abstract
            FROM articles a
            WHERE a.embedding IS NOT NULL
            ORDER BY (sub_vector(a.embedding, {dims})::vector({dims}) <=> sub_vector(%(query_embedding)s::vector, {dims})::vector({dims})) ASC
            LIMIT 25;
        """
        cursor.execute(sql_initial_search, {'query_embedding': query_embedding})
        initial_candidates = cursor.fetchall()

    if not initial_candidates:
        return []

    # === STEP 3: Rerank the Top 100 Candidates ===
    # We pass the original user query (without prefix) to the reranker
    reranked_results = await rerank_documents(query_text, initial_candidates)
    
    top_reranked_ids = [result['id'] for result in reranked_results]
    score_map = {result['id']: result['score'] for result in reranked_results}
    
    # === STEP 4: Fetch Author and Article Data for Aggregation ===
    author_contributions = []
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        sql_fetch_details = """
            SELECT
                au.author_id, au.full_name, au.scopus_id,
                a.article_id, a.title, a.year, a.source_title, a.link, a.abstract,
                aa.author_order
            FROM articles a
            JOIN articles_authors aa ON a.article_id = aa.article_id
            JOIN authors au ON aa.author_id = au.author_id
            WHERE a.article_id = ANY(%(article_ids)s);
        """
        cursor.execute(sql_fetch_details, {'article_ids': top_reranked_ids})
        author_contributions = cursor.fetchall()

    # === STEP 5: Aggregate Scores in Python ===
    expert_scores = defaultdict(float)
    expert_articles = defaultdict(list)
    expert_info = {}

    for contrib in author_contributions:
        author_id = contrib['author_id']
        article_id = contrib['article_id']
        
        # Store author info once
        if author_id not in expert_info:
            expert_info[author_id] = {
                'author_id': author_id,
                'full_name': contrib['full_name'],
                'scopus_id': contrib['scopus_id']
            }
        
        # Calculate weighted score based on author order and rerank score
        normalized_score = score_map.get(article_id, 0)
        author_order_weight = 1.0
        order = contrib['author_order']
        if order == 2: author_order_weight = 0.8
        elif order == 3: author_order_weight = 0.6
        elif order > 3: author_order_weight = 0.4
        
        weighted_score = normalized_score * author_order_weight
        expert_scores[author_id] += weighted_score

        # Append article details to the author's list
        expert_articles[author_id].append({
            'article_id': article_id,
            'title': contrib['title'],
            'year': contrib['year'],
            'source_title': contrib['source_title'],
            'link': contrib['link'],
            'similarity_score': normalized_score, # Use the reranked score
            'abstract': contrib['abstract']
        })

    # === STEP 6: Format Final Results ===
    final_results = []
    for author_id, total_score in expert_scores.items():
        # Sort articles for each expert by similarity score
        sorted_articles = sorted(expert_articles[author_id], key=lambda x: x['similarity_score'], reverse=True)
        
        author_data = expert_info[author_id]
        author_data['expert_score'] = total_score
        author_data['articles'] = sorted_articles
        final_results.append(author_data)

    # Sort experts by their total score and take the top 5
    final_results.sort(key=lambda x: x['expert_score'], reverse=True)
    
    return final_results[:5]


async def run_pairwise_search(query_text: str):
    """
    Orchestrates a pairwise search for experts.
    1. Fetches all available models from the DB.
    2. Randomly selects two different models.
    3. Concurrently runs find_experts for each model.
    4. Returns a structured dictionary with both results.
    """
    conn = get_db()
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT model_id, model_name FROM models;")
        all_models = cursor.fetchall()

    if len(all_models) < 2:
        raise ValueError("Not enough models configured for a pairwise test. At least 2 are required.")
    
    model_a_data, model_b_data = random.sample(all_models, 2)
    
    try:
        dims_a = int(model_a_data['model_name'].split('_')[0])
        dims_b = int(model_b_data['model_name'].split('_')[0])
    except (ValueError, IndexError):
        raise ValueError("Model name format is incorrect. Expected format like '256_dim'.")

    results_a_task = find_experts(query_text, dims_a)
    results_b_task = find_experts(query_text, dims_b)
    
    results_a, results_b = await asyncio.gather(results_a_task, results_b_task)

    return {
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