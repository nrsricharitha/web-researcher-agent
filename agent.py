import os
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Import LangGraph components
from langgraph.graph import StateGraph, END

# Import LangChain models
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import our custom types and tools
from state import AgentState
from tools import web_search, scrape_webpage

# Load environment variables
load_dotenv()

# Initialize LLM dynamically based on user provider selection
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in your .env file or Streamlit sidebar.")
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.2
        )
        
    elif provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is not set. Please configure it in your .env file or Streamlit sidebar.")
        model_name = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            model=model_name,
            temperature=0.2
        )
        
    else:  # Default to Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please configure it in your .env file or Streamlit sidebar.")
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.2,
            max_tokens=4096
        )

def clean_json_response(content: str) -> Dict[str, Any]:
    """
    Cleans reasoning thoughts (<think>...</think>) and markdown code blocks
    (like ```json ... ```) from LLM response and parses it into a dictionary.
    """
    # 1. Strip reasoning thoughts if present (common in Qwen/DeepSeek thinking models)
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    
    # 2. Remove markdown code blocks if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    
    # Try parsing
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback regex to find anything between { and }
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from content: {content}")

# ----------------- GRAPH NODES -----------------

def generate_queries_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates search queries based on the user's research task
    and what we have scraped so far.
    """
    task = state["task"]
    scraped_data = state.get("scraped_data", {})
    logs = list(state.get("logs", []))
    
    new_log = "🤖 Planning search strategy..."
    logs.append(new_log)
    print(new_log)
    
    llm = get_llm()
    
    # Context if we already have some data
    current_knowledge = ""
    if scraped_data:
        current_knowledge = "Currently scraped URLs and summaries:\n"
        for url, content in scraped_data.items():
            current_knowledge += f"- {url}: {content[:300]}...\n\n"
            
    system_prompt = (
        "You are an expert research planner. Your task is to generate up to 3 highly specific "
        "search engine queries that will help gather information for the user's research topic.\n"
        "If some information has already been gathered, focus your new queries on filling in "
        "missing details, verifying claims, or getting deeper information.\n\n"
        "You must respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "queries": ["query 1", "query 2", "query 3"]\n'
        "}"
    )
    
    user_prompt = f"User Request: {task}\n\n{current_knowledge}\nGenerate the next queries now."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    data = clean_json_response(response.content)
    queries = data.get("queries", [task])
    
    formatted_queries = ", ".join(f'"{q}"' for q in queries)
    query_logs = f"🔍 Generated search queries: {formatted_queries}"
    logs.append(query_logs)
    print(query_logs)
    
    return {
        "queries": queries,
        "logs": logs
    }

def search_web_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes search queries and gathers top URLs.
    """
    queries = state["queries"]
    existing_urls = set(state.get("urls", []))
    logs = list(state.get("logs", []))
    
    new_log = "🌐 Searching the live web..."
    logs.append(new_log)
    print(new_log)
    
    new_urls = []
    for query in queries:
        search_log = f"🔎 Searching for: \"{query}\""
        logs.append(search_log)
        print(search_log)
        
        results = web_search(query, max_results=3)
        for r in results:
            url = r.get("url")
            if url and url not in existing_urls:
                new_urls.append(url)
                existing_urls.add(url)
                
    url_log = f"🔗 Discovered {len(new_urls)} new URLs to analyze."
    logs.append(url_log)
    print(url_log)
    
    return {
        "urls": list(existing_urls),
        "logs": logs
    }

def scrape_pages_node(state: AgentState) -> Dict[str, Any]:
    """
    Scrapes the newly gathered URLs.
    """
    urls = state["urls"]
    scraped_data = dict(state.get("scraped_data", {}))
    logs = list(state.get("logs", []))
    
    # Identify which URLs we haven't scraped yet
    urls_to_scrape = [url for url in urls if url not in scraped_data]
    
    # Limit scraping to top 2 pages per loop to prevent context bloat
    urls_to_scrape = urls_to_scrape[:2]
    
    if not urls_to_scrape:
        no_scrape_log = "💡 No new URLs to scrape in this loop."
        logs.append(no_scrape_log)
        print(no_scrape_log)
        return {"logs": logs}
        
    scrape_log = f"📖 Scraping {len(urls_to_scrape)} webpages..."
    logs.append(scrape_log)
    print(scrape_log)
    
    for url in urls_to_scrape:
        page_log = f"📄 Extracting content from: {url}"
        logs.append(page_log)
        print(page_log)
        
        content = scrape_webpage(url)
        scraped_data[url] = content
        
    return {
        "scraped_data": scraped_data,
        "logs": logs
    }

def evaluate_info_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether we have enough information to write the report,
    or if we need to search more.
    """
    task = state["task"]
    scraped_data = state.get("scraped_data", {})
    current_loop = state.get("current_loop", 0)
    logs = list(state.get("logs", []))
    
    new_log = "🧠 Analyzing information completeness..."
    logs.append(new_log)
    print(new_log)
    
    # If we have no scraped data, we definitely need to search
    if not scraped_data:
        logs.append("⚠️ No data scraped yet. Continuing research.")
        return {
            "next_action": "continue",
            "current_loop": current_loop + 1,
            "logs": logs
        }
        
    # Prevent infinite loops
    max_loops = 3
    if current_loop >= max_loops:
        logs.append(f"⏱️ Loop limit ({max_loops}) reached. Forcing report compilation.")
        return {
            "next_action": "write",
            "logs": logs
        }
        
    llm = get_llm()
    
    # Compile gathered text for analysis
    compiled_knowledge = ""
    for url, text in scraped_data.items():
        compiled_knowledge += f"Source URL: {url}\nContent Snippet:\n{text[:1500]}...\n\n"
        
    system_prompt = (
        "You are a critical quality control research analyst. Your job is to evaluate if the "
        "accumulated web research content is sufficient to write a deep, detailed, and comprehensive "
        "report about the user's topic.\n"
        "Check for completeness, facts, and whether the primary questions are answered.\n\n"
        "Respond ONLY with a JSON object in this format:\n"
        "{\n"
        '  "enough_info": true or false,\n'
        '  "reason": "explanation of why it is sufficient or what specifically is still missing"\n'
        "}"
    )
    
    user_prompt = f"Topic: {task}\n\nGathered Research:\n{compiled_knowledge}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        data = clean_json_response(response.content)
        enough_info = data.get("enough_info", True)
        reason = data.get("reason", "")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        enough_info = True
        reason = "Error in evaluation, proceeding to write."
        
    eval_log = f"📊 Evaluation: Enough Info = {enough_info}. Reason: {reason}"
    logs.append(eval_log)
    print(eval_log)
    
    next_action = "write" if enough_info else "continue"
    
    return {
        "next_action": next_action,
        "current_loop": current_loop + 1,
        "logs": logs
    }

def write_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Compiles and synthesizes all scraped data into a comprehensive Markdown report,
    and saves it to a file.
    """
    task = state["task"]
    scraped_data = state.get("scraped_data", {})
    logs = list(state.get("logs", []))
    
    new_log = "✍️ Compiling final report..."
    logs.append(new_log)
    print(new_log)
    
    # If using Groq, pause for 10 seconds to allow the rolling TPM rate-limit window to decay
    if os.getenv("LLM_PROVIDER", "groq").lower() == "groq":
        time.sleep(10)
    
    llm = get_llm()
    
    # Format all scraped content for LLM synthesis (trimmed to 3500 chars to fit Groq free tier rate limits)
    full_contexts = ""
    for url, text in scraped_data.items():
        trimmed_text = text[:2000] if text else "No content scraped."
        full_contexts += f"--- SOURCE: {url} ---\n{trimmed_text}\n\n"
        
    system_prompt = (
        "You are an expert technical writer and researcher. Your task is to write a comprehensive, "
        "highly detailed, and professional research report based ONLY on the provided scraped content.\n"
        "Guidelines:\n"
        "- Do not hallucinate or add facts not present in the sources.\n"
        "- Use markdown formatting (headings, bullet points, code blocks where relevant, bold text).\n"
        "- Include an Executive Summary, Detailed Analysis by Subtopics, and a 'Sources Cited' "
        "section listing all original URLs used in your research.\n"
        "- The report must be clear, well-structured, and easy to read."
    )
    
    user_prompt = f"Topic to Research: {task}\n\nScraped Web Sources:\n{full_contexts}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    report = response.content
    
    # Save the report to the output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create file name based on task
    safe_title = re.sub(r'[^a-zA-Z0-9]', '_', task)[:30].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"report_{safe_title}_{timestamp}.md")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    success_log = f"💾 Report saved successfully to: {file_path}"
    logs.append(success_log)
    print(success_log)
    
    return {
        "report": report,
        "logs": logs
    }

# ----------------- CONDITIONAL ROUTING -----------------

def should_continue(state: AgentState) -> str:
    """
    Decides whether to route back to search or end at report writing.
    """
    action = state.get("next_action", "write")
    if action == "continue":
        return "continue"
    else:
        return "end"

# ----------------- GRAPH COMPILATION -----------------

def build_agent_graph():
    """
    Creates and compiles the LangGraph StateGraph.
    """
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("scrape_pages", scrape_pages_node)
    workflow.add_node("evaluate_info", evaluate_info_node)
    workflow.add_node("write_report", write_report_node)
    
    # Connect the nodes
    workflow.set_entry_point("generate_queries")
    workflow.add_edge("generate_queries", "search_web")
    workflow.add_edge("search_web", "scrape_pages")
    workflow.add_edge("scrape_pages", "evaluate_info")
    
    # Routing decision
    workflow.add_conditional_edges(
        "evaluate_info",
        should_continue,
        {
            "continue": "generate_queries",
            "end": "write_report"
        }
    )
    
    workflow.add_edge("write_report", END)
    
    return workflow.compile()

# Instantiated graph for import
research_agent = build_agent_graph()
