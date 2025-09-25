import asyncio
import aiohttp
import os
from typing import List, AsyncIterator, Tuple, Iterable, Dict, Any, Callable

def _create_chunks(data: List[Any], size: int) -> Iterable[List[Any]]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def _get_endpoint_and_headers(service: str) -> Tuple[str, dict]:
    """
    Gets the appropriate URL and headers for a given service ('embedding' or 'rerank').
    This centralizes service configuration.
    """
    if service == 'embedding':
        mode = os.environ.get("EMBEDDING_MODE", "local").lower()
        if mode == "local":
            endpoint_url = os.environ.get("LOCAL_TEI_URL", "http://tei-embedding:80/embed")
            headers = {"Content-Type": "application/json"}
        elif mode == "hf":
            api_token = os.environ.get("HF_API_TOKEN")
            endpoint_url = os.environ.get("HF_INFERENCE_ENDPOINT_URL")
            if not api_token or not endpoint_url:
                raise ValueError("HF_API_TOKEN and HF_INFERENCE_ENDPOINT_URL must be set for hf mode.")
            headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        else:
            raise ValueError(f"Unsupported EMBEDDING_MODE: {mode}")
        return endpoint_url, headers
    
    elif service == 'rerank':
        endpoint_url = "http://tei-reranker:80/rerank"
        headers = {"Content-Type": "application/json"}
        return endpoint_url, headers

    else:
        raise ValueError(f"Unknown service type: {service}")


async def _process_batch_concurrently(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    batch: Any,
    payload_builder: Callable[[Any], Dict],
    response_processor: Callable[[Any, Any], Any],
    semaphore: asyncio.Semaphore
) -> Any:
    """
    A generic coroutine to process a single batch. It builds the payload, makes the request,
    processes the response, and manages semaphore access.
    """
    async with semaphore:
        payload = payload_builder(batch)
        async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
            resp.raise_for_status()
            response_json = await resp.json()
            return response_processor(batch, response_json)


# ==============================================================================
# Public API Functions
# ==============================================================================

async def get_embeddings(
    docs_with_ids: Iterable[Tuple[List, List[str]]],
    concurrency: int
) -> AsyncIterator[Tuple[List, List[List[float]]]]:
    """
    Generates embeddings for batches of documents streamed from an iterable.
    This function's external behavior is unchanged.
    """
    endpoint_url, headers = _get_endpoint_and_headers('embedding')
    sem = asyncio.Semaphore(concurrency)

    def build_payload(batch: Tuple[List, List[str]]) -> Dict:
        _, documents = batch
        return {"inputs": documents}

    def process_response(batch: Tuple[List, List[str]], response: List[List[float]]) -> Tuple[List, List[List[float]]]:
        article_ids, _ = batch
        return article_ids, response

    async with aiohttp.ClientSession() as session:
        tasks = set()
        for batch in docs_with_ids:
            task = asyncio.create_task(
                _process_batch_concurrently(session, endpoint_url, headers, batch, build_payload, process_response, sem)
            )
            tasks.add(task)
            if len(tasks) >= concurrency * 2:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for done_task in done:
                    yield await done_task
        
        if tasks:
            for task in asyncio.as_completed(tasks):
                yield await task


async def rerank_documents(query: str, documents: List[Dict]) -> List[Dict]:
    RERANKER_BATCH_SIZE = 8
    CONCURRENCY = 16
    
    endpoint_url, headers = _get_endpoint_and_headers('rerank')
    sem = asyncio.Semaphore(CONCURRENCY)

    def build_payload(batch: List[Dict]) -> Dict:
        texts = [f"{doc.get('title', '')}\n{doc.get('abstract', '')}".strip() for doc in batch]
        return {"query": query, "texts": texts, "return_documents": False}

    def process_response(batch: List[Dict], response: List[Dict]) -> List[Dict]:
        return [
            {'id': batch[result['index']]['article_id'], 'score': result['score']}
            for result in response
        ]

    all_scored_articles = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for batch in _create_chunks(documents, RERANKER_BATCH_SIZE):
            task = asyncio.create_task(
                _process_batch_concurrently(session, endpoint_url, headers, batch, build_payload, process_response, sem)
            )
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            all_scored_articles.extend(result)

    all_scored_articles.sort(key=lambda x: x['score'], reverse=True)
    return all_scored_articles


async def get_single_embedding(text: str) -> List[float]:
    endpoint_url, headers = _get_endpoint_and_headers('embedding')
    payload = {"inputs": [text]}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint_url, json=payload, headers=headers, timeout=60) as resp:
            resp.raise_for_status()
            embeddings = await resp.json()
            if isinstance(embeddings, list) and len(embeddings) > 0:
                return embeddings[0]
            else:
                raise ValueError("Failed to get a valid embedding from the API.")