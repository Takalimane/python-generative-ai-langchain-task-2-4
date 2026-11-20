from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

from src.rag_agent_app.agent import build_agent
from src.rag_agent_app.config import AppConfig
from src.rag_agent_app.providers import build_chat_model, build_embeddings
from src.rag_agent_app.rag import DocumentIndex


load_dotenv()
st.set_page_config(page_title="Grounded RAG Agent", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_index" not in st.session_state:
    st.session_state.document_index = None
if "provider" not in st.session_state:
    st.session_state.provider = os.getenv("AI_PROVIDER", "openai").lower()

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.65rem; }
    [data-testid="stChatMessage"] { border: 1px solid #eef2f7; border-radius: 16px; margin-bottom: 8px; }
    [data-testid="stChatInput"] textarea { border-radius: 16px !important; }
    .app-header { display: flex; align-items: center; min-height: 64px; margin-bottom: 4px; }
    .app-title { font-size: 1.35rem; font-weight: 700; color: #0f172a; }
    .app-subtitle { color: #64748b; font-size: 0.88rem; }
    .welcome-card { max-width: 720px; margin: 18vh auto 0; text-align: center; }
    .welcome-card h2 { color: #0f172a; }
    .welcome-card p { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([9, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        '<div class="app-header"><div><div class="app-title">Document Assistant</div>'
        '<div class="app-subtitle">Grounded RAG, research tools, and conversation</div></div></div>',
        unsafe_allow_html=True,
    )
with header_right:
    with st.popover("⚙️  Settings", use_container_width=True):
        st.subheader("AI configuration")
        provider = st.selectbox(
            "Provider",
            options=["openai", "gemini"],
            index=0 if st.session_state.provider == "openai" else 1,
        )
        st.session_state.provider = provider
        os.environ["AI_PROVIDER"] = provider

        if provider == "openai":
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-...",
                help="Used only by this running session.",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            os.environ["EMBEDDING_PROVIDER"] = "openai"
        else:
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                placeholder="Enter API key",
                help="Used only by this running session.",
            )
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            os.environ["EMBEDDING_PROVIDER"] = "gemini"

        tavily_key = st.text_input(
            "Tavily API key (optional)",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            help="Needed only when web search is enabled.",
        )
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        required_key = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        if os.getenv(required_key):
            st.success("API key connected")
        else:
            st.info("Configure an API key to begin.")

with st.sidebar:
    st.markdown("### **RAG Workspace**")
    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.document_index = None
        st.rerun()

    st.markdown("#### Documents")
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Index documents", use_container_width=True):
        try:
            config = AppConfig.from_env()
            index = DocumentIndex(build_embeddings(config))
            saved_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded:
                    path = os.path.join(temp_dir, file.name)
                    with open(path, "wb") as handle:
                        handle.write(file.getbuffer())
                    saved_paths.append(path)
                count = index.load_files(saved_paths)
            st.session_state.document_index = index
            st.success(f"Indexed {count} document chunks.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.markdown("#### Tools")
    enable_web = st.toggle("Enable web search", value=False)

    if st.session_state.document_index is not None:
        st.caption("✓ Documents indexed")
    else:
        st.caption("No documents indexed")

    st.markdown("#### Recent")
    user_messages = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_messages:
        for recent in user_messages[-4:]:
            st.caption(recent[:48] + ("…" if len(recent) > 48 else ""))
    else:
        st.caption("No recent conversations")

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
        <h2>What would you like to explore?</h2>
        <p>Upload documents, ask a grounded question, calculate an expression, or enable web search.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

question = st.chat_input("Ask about your documents or use a tool...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response_box = st.empty()
        try:
            config = AppConfig.from_env()
            required_key = "OPENAI_API_KEY" if config.provider == "openai" else "GEMINI_API_KEY"
            if not os.getenv(required_key):
                raise ValueError(
                    f"Open Settings and enter a {config.provider.title()} API key to start chatting."
                )
            chat_model = build_chat_model(config)

            rag_tool = None
            index = st.session_state.document_index
            if index is not None:
                def search_documents(query: str) -> str:
                    chunks = index.retrieve(query)
                    if not chunks:
                        return "No relevant document content found."
                    parts = []
                    for chunk in chunks:
                        source = chunk.metadata.get("source", "unknown")
                        page = chunk.metadata.get("page")
                        citation = f"{source}, page {page + 1}" if page is not None else source
                        parts.append(f"[Source: {citation}] {chunk.page_content}")
                    return "\n\n".join(parts)

                rag_tool = StructuredTool.from_function(
                    func=search_documents,
                    name="document_search",
                    description="Search uploaded PDF or text documents and return relevant passages with citations.",
                )

            web_tool = None
            if enable_web:
                if not config.tavily_key:
                    raise ValueError("TAVILY_API_KEY is required for web search")
                from langchain_community.tools.tavily_search import TavilySearchResults

                web_tool = TavilySearchResults(
                    max_results=5,
                    tavily_api_key=config.tavily_key,
                )

            executor = build_agent(chat_model, web_tool, rag_tool)
            result = executor.invoke({
                "input": question,
                "chat_history": [
                    {"role": message["role"], "content": message["content"]}
                    for message in st.session_state.messages[:-1]
                ],
            })
            answer = result["output"]
        except Exception as exc:
            answer = f"Error: {exc}"

        response_box.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
