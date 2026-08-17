import os
import streamlit as st
import time
from dotenv import load_dotenv

# Import state and agent
from state import AgentState
from agent import research_agent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Web Researcher Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .report-container {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-top: 20px;
        color: #e6edf3;
    }
    .log-box {
        background-color: #161b22;
        color: #e6edf3;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4f46e5;
        font-family: monospace;
        font-size: 0.9rem;
        margin-bottom: 15px;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)



# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ Configurations")

# LLM Provider Selection
st.sidebar.subheader("🤖 LLM Provider Selection")
provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    options=["Groq", "OpenAI", "NVIDIA NIM"],
    index=0
)
provider_lower = provider.lower()
if provider_lower == "nvidia nim":
    provider_lower = "nvidia"
os.environ["LLM_PROVIDER"] = provider_lower

# API Keys Configuration
st.sidebar.subheader("🔑 API Keys")
env_tavily_key = os.getenv("TAVILY_API_KEY")
tavily_placeholder = "Key configured via .env 🔒" if env_tavily_key else "tvly_..."

# Show API Key based on provider
if provider_lower == "groq":
    env_groq_key = os.getenv("GROQ_API_KEY")
    groq_placeholder = "Key configured via .env 🔒" if env_groq_key else "gsk_..."
    groq_input = st.sidebar.text_input(
        "Groq API Key",
        value="",
        placeholder=groq_placeholder,
        type="password",
        help="If left blank, uses the key from your .env file."
    )
    if groq_input.strip():
        os.environ["GROQ_API_KEY"] = groq_input.strip()
    elif env_groq_key:
        os.environ["GROQ_API_KEY"] = env_groq_key

elif provider_lower == "openai":
    env_openai_key = os.getenv("OPENAI_API_KEY")
    openai_placeholder = "Key configured via .env 🔒" if env_openai_key else "sk-..."
    openai_input = st.sidebar.text_input(
        "OpenAI API Key",
        value="",
        placeholder=openai_placeholder,
        type="password",
        help="If left blank, uses the key from your .env file."
    )
    if openai_input.strip():
        os.environ["OPENAI_API_KEY"] = openai_input.strip()
    elif env_openai_key:
        os.environ["OPENAI_API_KEY"] = env_openai_key

elif provider_lower == "nvidia":
    env_nvidia_key = os.getenv("NVIDIA_API_KEY")
    nvidia_placeholder = "Key configured via .env 🔒" if env_nvidia_key else "nvapi-..."
    nvidia_input = st.sidebar.text_input(
        "NVIDIA API Key",
        value="",
        placeholder=nvidia_placeholder,
        type="password",
        help="If left blank, uses the key from your .env file."
    )
    if nvidia_input.strip():
        os.environ["NVIDIA_API_KEY"] = nvidia_input.strip()
    elif env_nvidia_key:
        os.environ["NVIDIA_API_KEY"] = env_nvidia_key

tavily_input = st.sidebar.text_input(
    "Tavily API Key (Optional)",
    value="",
    placeholder=tavily_placeholder,
    type="password",
    help="Optional. If left blank, agent will use free DuckDuckGo search."
)
if tavily_input.strip():
    os.environ["TAVILY_API_KEY"] = tavily_input.strip()
elif env_tavily_key:
    os.environ["TAVILY_API_KEY"] = env_tavily_key

# LLM Model Selection
st.sidebar.subheader("📦 Model Selection")
if provider_lower == "groq":
    models = [
        "qwen/qwen3.6-27b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it"
    ]
elif provider_lower == "openai":
    models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo"
    ]
else:  # nvidia nim
    models = [
        "meta/llama-3.2-3b-instruct",
        "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.1-405b-instruct"
    ]

model_selection = st.sidebar.selectbox(
    "Choose Model",
    options=models,
    index=0,
    help="Select the LLM model to power the agent."
)
os.environ["LLM_MODEL"] = model_selection

# Check Status
st.sidebar.subheader("🚦 Provider Status")
if provider_lower == "groq":
    if os.getenv("GROQ_API_KEY"):
        st.sidebar.success("🟢 Groq API Key Active")
    else:
        st.sidebar.warning("🔴 Groq API Key Missing")
elif provider_lower == "openai":
    if os.getenv("OPENAI_API_KEY"):
        st.sidebar.success("🟢 OpenAI API Key Active")
    else:
        st.sidebar.warning("🔴 OpenAI API Key Missing")
elif provider_lower == "nvidia":
    if os.getenv("NVIDIA_API_KEY"):
        st.sidebar.success("🟢 NVIDIA API Key Active")
    else:
        st.sidebar.warning("🔴 NVIDIA API Key Missing")

if os.getenv("TAVILY_API_KEY"):
    st.sidebar.info("🟢 Tavily Search Active")
else:
    st.sidebar.info("ℹ️ DuckDuckGo Search (Free) Active")

# ----------------- MAIN UI -----------------
st.title("🔍 Autonomous Web Researcher & Reporter")
st.markdown(
    "Submit a research topic, and watch this LangGraph agent autonomously search the web, "
    "scrape sources, evaluate data completeness, and draft a professional report."
)

# User input
user_query = st.text_area(
    "What topic would you like me to research?",
    placeholder="Example: Latest developments in room-temperature superconductors in 2026",
    height=100
)

# Run button
if st.button("🚀 Start Autonomous Research", use_container_width=True):
    missing_key = False
    if provider_lower == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error("Please provide a Groq API Key in the sidebar or in a .env file to run the agent.")
        missing_key = True
    elif provider_lower == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API Key in the sidebar or in a .env file to run the agent.")
        missing_key = True
    elif provider_lower == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
        st.error("Please provide an NVIDIA API Key in the sidebar or in a .env file to run the agent.")
        missing_key = True

    if not missing_key:
        if not user_query.strip():
            st.warning("Please enter a research topic to start.")
        else:
            st.subheader("🤖 Agent Execution Logs")
        
        # Placeholders for UI updating
        log_placeholder = st.empty()
        status_placeholder = st.info("Initializing Agent...")
        
        # Initialize graph state
        initial_state: AgentState = {
            "task": user_query.strip(),
            "queries": [],
            "urls": [],
            "scraped_data": {},
            "current_loop": 0,
            "next_action": "continue",
            "report": "",
            "logs": ["🚀 Agent initialized. Task received."]
        }
        
        # Run graph execution loop with streaming
        try:
            final_state = initial_state
            
            # Streaming events from LangGraph
            for event in research_agent.stream(initial_state):
                for node_name, output in event.items():
                    # Merge outputs into final_state to keep tracking latest state
                    for key, val in output.items():
                        final_state[key] = val
                    
                    # Update status
                    status_placeholder.info(f"Agent executing node: **{node_name.upper()}**")
                    
                    # Render updated logs list inside box
                    with log_placeholder.container():
                        log_html = "<div class='log-box'>"
                        for log in final_state["logs"]:
                            log_html += f"<div>{log}</div>"
                        log_html += "</div>"
                        st.markdown(log_html, unsafe_allow_html=True)
                        
                time.sleep(0.5)  # Slight delay to make logs readable as they stream
            
            # Finished state
            status_placeholder.success("🎯 Research complete! Report generated successfully.")
            
            # Display Report
            st.divider()
            st.subheader("📄 Generated Report")
            
            report_content = final_state.get("report", "Error: No report was generated.")
            
            st.markdown(
                f"<div class='report-container'>{report_content}</div>",
                unsafe_allow_html=True
            )
            
            # Download button
            st.download_button(
                label="📥 Download Markdown Report",
                data=report_content,
                file_name=f"research_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        except Exception as e:
            status_placeholder.error(f"An error occurred during execution: {str(e)}")
            st.exception(e)
