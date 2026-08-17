import os
from typing import List, Dict, Any
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a web search using Tavily API (if key exists) or DuckDuckGo (fallback).
    Returns a list of dicts with keys: 'title', 'url', and 'snippet'.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            import requests
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": query,
                "max_results": max_results
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                results = response.json()
                formatted = []
                for r in results.get("results", []):
                    formatted.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")
                    })
                return formatted
            else:
                print(f"Tavily API error: Status {response.status_code}")
        except Exception as e:
            print(f"Tavily API request failed: {e}")
            
    # Free DuckDuckGo fallback
    try:
        formatted = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if results:
                for r in results:
                    formatted.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        return formatted
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return []
