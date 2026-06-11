import json
import os
import time

from google import genai
from google.genai import types

# Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "gemma-4-31b-it"


def format_experts_for_prompt(experts_list):
    """Formats the expert list into a readable string for the LLM."""
    formatted = ""
    for i, expert in enumerate(experts_list[:3]):
        formatted += f"Expert {i + 1}: {expert['full_name']}\n"

        for article in expert["articles"][:3]:
            formatted += f"  - [{article['year']}] {article['title']}\n"
            formatted += f"    Citations: {article['cited_by']}\n"
            formatted += f"    Abstract snippet: {article['abstract'][:300]}...\n"
        formatted += "\n"
    return formatted


def evaluate_with_gemini(query, results_a, results_b):
    """Sends the data to Gemini and forces a JSON response."""

    prompt = f"""
    You are a Senior Academic Evaluator. Your task is to evaluate two lists of experts retrieved for a specific research query.

    Research Query: "{query}"

    === LIST A ===
    {format_experts_for_prompt(results_a)}

    === LIST B ===
    {format_experts_for_prompt(results_b)}

    EVALUATION CRITERIA (Evaluate in order of priority):

    1. DIRECT SEMANTIC RELEVANCE (Highest Priority)
       - Look at the article titles and abstracts for the experts in both lists.
       - A list is BETTER if its experts have published papers that directly and specifically address the query topic.
       - Penalize lists containing "generalists" whose papers only tangentially mention the topic compared to "specialists" whose papers are highly focused on the query.

    2. TOP-HEAVY ALIGNMENT
       - Pay closest attention to the Expert #1 and Expert #2 spots.
       - The absolute best, most highly-specific expert must be at the very top of the list (Rank 1).
       - If List A has a better Rank 1 expert than List B, List A is the winner, even if List B's 3rd expert is slightly better.

    3. ACADEMIC AUTHORITY & CITATION WEIGHT (Tie-Breaker)
       - If both lists contain experts with similar semantic relevance, look at the citation counts ("Citations:") of their papers.
       - An expert with highly cited papers in the query's niche represents greater academic authority and is a better recommendation.

    4. COHERENCE OF THE LIST
       - Evaluate if all 3 experts belong to the same scientific community related to the query, or if the list contains random, unrelated authors. A coherent list of peers is better.

    DECISION LOGIC:
    - Choose 'a' if List A is superior based on the criteria above.
    - Choose 'b' if List B is superior based on the criteria above.
    - Choose 'draw' ONLY if:
      a) Both lists are of equally high, excellent quality.
      b) Both lists are completely irrelevant to the query (i.e., no experts in either list have any relation to the query topic).

    OUTPUT FORMAT:

    You MUST respond in strict JSON format like this:
    {{"choice": "a", "reasoning": "Brief explanation of why A is better."}}
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.1
                ),
            )
            result = json.loads(response.text)

            # Validate output
            if result.get("choice") not in ["a", "b", "draw"]:
                result["choice"] = "draw"

            return result

        except Exception as e:
            if "429" in str(e):
                print(
                    f"  !! Rate limit hit. Waiting 60 seconds... (Attempt {attempt + 1})"
                )
                time.sleep(60)
            else:
                print(f"  !! API Error: {e}. Waiting 10s...")
                time.sleep(10)

    return {"choice": None, "reasoning": "API failed after multiple retries."}
