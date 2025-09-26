from db import get_db
from vectorizer import get_single_embedding, rerank_documents
from psycopg2.extras import RealDictCursor
import asyncio
import random
from collections import defaultdict

async def find_experts(
    query_text: str,
    initial_dims: int,
    rerank_method: str = 'cross-encoder',
    shortlist_size: int = 25
):
    """
    Orchestrates a two-stage process of finding and ranking experts.

    Args:
        query_text (str): The user's search query.
        initial_dims (int): The vector dimension for the fast initial retrieval.
        rerank_method (str): The method for the second stage ('cross-encoder' or 'vector').
        shortlist_size (int): The number of candidates for the second stage.
    """
    conn = get_db()
    
    # === STEP 1: Get Query Embedding ===
    prefixed_query = f"search_query: {query_text}"
    query_embedding = await get_single_embedding(prefixed_query)

    # === STEP 2: Initial Candidate Retrieval (Stage 1) ===
    # Fast retrieval using a lower-dimension index to get a shortlist of IDs.
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        sql_initial_search = f"""
            SELECT a.article_id, a.title, a.abstract
            FROM articles a
            WHERE a.embedding IS NOT NULL
            ORDER BY (sub_vector(a.embedding, %(initial_dims)s)::vector(%(initial_dims)s) <=> sub_vector(%(query_embedding)s::vector, %(initial_dims)s)::vector(%(initial_dims)s)) ASC
            LIMIT %(shortlist_size)s;
        """
        cursor.execute(sql_initial_search, {
            'query_embedding': query_embedding,
            'initial_dims': initial_dims,
            'shortlist_size': shortlist_size
        })
        initial_candidates = cursor.fetchall()

    if not initial_candidates:
        return []

    # === STEP 3: Reranking (Stage 2) ===
    score_map = {}
    top_ranked_ids = []

    if rerank_method == 'cross-encoder':
        # Rerank the small shortlist using the powerful cross-encoder model.
        reranked_results = await rerank_documents(query_text, initial_candidates)
        top_ranked_ids = [result['id'] for result in reranked_results]
        score_map = {result['id']: result['score'] for result in reranked_results}
    
    elif rerank_method == 'vector':
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
    else:
        raise ValueError(f"Unknown rerank_method: '{rerank_method}'")
    
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
        cursor.execute(sql_fetch_details, {'article_ids': top_ranked_ids})
        author_contributions = cursor.fetchall()

    # === STEP 5: Aggregate Scores in Python ===
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
            'abstract': contrib['abstract']
        })

    # === STEP 6: Format Final Results ===
    final_results = []
    for author_id, total_score in expert_scores.items():
        sorted_articles = sorted(expert_articles[author_id], key=lambda x: x['similarity_score'], reverse=True)
        
        author_data = expert_info[author_id]
        author_data['expert_score'] = total_score
        author_data['articles'] = sorted_articles
        final_results.append(author_data)

    final_results.sort(key=lambda x: x['expert_score'], reverse=True)
    
    return final_results[:5]


async def run_pairwise_search(query_text: str):
    """
    Orchestrates a pairwise search comparing two randomly selected retrieval models.
    Both models use the same powerful cross-encoder reranking stage to ensure
    that the primary variable being tested is the quality of the initial retrieval.
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

    task_a = find_experts(
        query_text=query_text,
        initial_dims=dims_a,
        rerank_method='vector',
        shortlist_size=100
    )
    task_b = find_experts(
        query_text=query_text,
        initial_dims=dims_b,
        rerank_method='vector',
        shortlist_size=100
    )
    
    results_a, results_b = await asyncio.gather(task_a, task_b)

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