import os
import json
import time
from google import genai
from google.genai import types

# Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Use the fast, free tier model
MODEL_NAME = "gemini-3.5-flash"

def format_experts_for_prompt(experts_list):
    """Formats the expert list into a readable string for the LLM."""
    formatted = ""
    for i, expert in enumerate(experts_list):
        formatted += f"Expert {i+1}: {expert['full_name']}\n"
        formatted += f"Total Expert Score: {expert['expert_score']:.2f}\n"
        for j, article in enumerate(expert['articles']):
            formatted += f"  - Article: {article['title']} ({article['year']})\n"
            formatted += f"    Citations: {article['cited_by']}\n"
            formatted += f"    Abstract: {article['abstract']}\n"
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
    
    Analyze the relevance of the articles to the query, the authority of the authors (citations), and the overall quality of the match.
    Which list is better? Choose 'a', 'b', or 'draw' if they are equally good.
    
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
                    response_mime_type='application/json',
                    temperature=0.1
                )
            )
            result = json.loads(response.text)
            
            # Validate output
            if result.get("choice") not in ['a', 'b', 'draw']:
                result["choice"] = "draw"
                
            return result
            
        except Exception as e:
            if "429" in str(e):
                print(f"  !! Rate limit hit. Waiting 60 seconds... (Attempt {attempt+1})")
                time.sleep(60)
            else:
                print(f"  !! API Error: {e}. Waiting 10s...")
                time.sleep(10)
            
    return {"choice": None, "reasoning": "API failed after multiple retries."}