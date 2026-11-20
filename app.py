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
st.title("📚 Document RAG & Tool Agent")
st.caption("Upload PDFs or text files, ask grounded questions, calculate, or search the web.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_index" not in st.session_state:
    st.session_state.document_index = None

with st.sidebar:
    st.header("Provider")
    provider = st.selectbox(
        "Chat provider",
        options=["openai", "gemini"],
        index=0 if os.getenv("AI_PROVIDER", "openai").lower() == "openai" else 1,
    )
    os.environ["AI_PROVIDER"] = provider

    if provider == "openai":
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Stored only in this running session.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        st.caption("Embedding provider: OpenAI")
        os.environ["EMBEDDING_PROVIDER"] = "openai"
    else:
        api_key = st.text_input(
            "Gemini API key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Stored only in this running session.",
        )
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        st.caption("Embedding provider: Gemini")
        os.environ["EMBEDDING_PROVIDER"] = "gemini"

    if not api_key:
        st.warning("Enter an API key to enable chat and document indexing.")

    st.header("Documents")
    uploaded = st.file_uploader(
        "Add PDF, TXT, MD, or CSV",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Build document index", use_container_width=True):
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

    st.header("Tools")
    enable_web = st.toggle("Enable web search", value=False)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

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
                    f"Enter a {config.provider.title()} API key in the sidebar to start chatting."
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
