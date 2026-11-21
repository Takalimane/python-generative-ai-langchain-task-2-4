from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.rag_agent_app.agent import build_agent
from src.rag_agent_app.config import AppConfig
from src.rag_agent_app.providers import build_chat_model, build_embeddings
from src.rag_agent_app.rag import DocumentIndex


load_dotenv()
st.set_page_config(
    page_title="Knowledge Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_index" not in st.session_state:
    st.session_state.document_index = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "provider" not in st.session_state:
    st.session_state.provider = os.getenv("AI_PROVIDER", "gemini").lower()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


st.markdown(
    """
    <style>
    :root {
      --ink: #18212f;
      --muted: #6d7788;
      --line: #e8ebf0;
      --surface: #ffffff;
      --surface-2: #f7f8fa;
      --brand: #5b5ce2;
      --brand-soft: #f0f0ff;
      --green: #1f9d78;
    }
    html, body, [class*="css"] {
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stHeader"] { height: 0; }
    [data-testid="stAppViewContainer"] { background: #fbfbfc; }
    [data-testid="stMainBlockContainer"] {
      max-width: 1180px;
      padding: 0 1.5rem 1.5rem 1.5rem;
    }
    [data-testid="stSidebar"] {
      background: #f3f4f6;
      border-right: 1px solid #e5e7eb;
      min-width: 248px;
    }
    [data-testid="stSidebar"] > div:first-child {
      padding-top: 1.1rem;
      padding-left: 1.15rem;
      padding-right: 1.15rem;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
    [data-testid="stChatInput"] {
      border: 1px solid #dfe3ea;
      border-radius: 18px;
      box-shadow: 0 8px 28px rgba(23, 31, 44, 0.08);
      background: white;
      max-width: 820px;
      margin: 0 auto;
    }
    [data-testid="stChatInput"] textarea {
      min-height: 52px;
      font-size: 0.95rem;
    }
    [data-testid="stChatMessage"] {
      border: 0;
      background: transparent;
      padding: 0.5rem 0;
    }
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
      background: var(--brand);
    }
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
      background: #202938;
    }
    .topbar {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: rgba(251, 251, 252, 0.96);
      margin: 0 -1.5rem 0 -1.5rem;
      padding: 0 1.5rem;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-mark {
      width: 34px; height: 34px; border-radius: 10px;
      background: linear-gradient(135deg, #6c70ef, #4547b8);
      color: white; display: grid; place-items: center;
      font-size: 17px; box-shadow: 0 6px 16px rgba(91,92,226,.22);
    }
    .brand-name { font-weight: 740; font-size: 1rem; letter-spacing: -0.02em; }
    .brand-status { color: var(--muted); font-size: 0.78rem; margin-top: 1px; }
    .status-dot {
      display: inline-block; width: 7px; height: 7px;
      border-radius: 50%; background: var(--green); margin-right: 5px;
    }
    .workspace-title {
      color: #424b5a;
      font-size: 0.72rem;
      font-weight: 750;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin: 0.2rem 0 0.6rem;
    }
    .side-section {
      border-top: 1px solid #e4e6ea;
      margin-top: 0.55rem;
      padding-top: 0.75rem;
    }
    .side-label {
      color: #7c8492;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.35rem;
    }
    .new-chat {
      border: 1px solid #dde1e7;
      background: white;
      border-radius: 9px;
    }
    .welcome {
      max-width: 720px;
      margin: 13vh auto 0;
      text-align: center;
    }
    .welcome-kicker {
      color: var(--brand);
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .welcome h1 {
      font-size: clamp(1.8rem, 4vw, 2.65rem);
      line-height: 1.08;
      letter-spacing: -0.045em;
      margin: 0.65rem 0 0.75rem;
    }
    .welcome p {
      color: var(--muted);
      max-width: 560px;
      margin: 0 auto 1.5rem;
      line-height: 1.65;
    }
    .prompt-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
    .source-panel {
      height: calc(100vh - 84px);
      border-left: 1px solid var(--line);
      padding: 1.15rem 0 1.15rem 1.15rem;
      position: sticky;
      top: 64px;
      overflow-y: auto;
    }
    .source-heading {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 0.8rem;
    }
    .source-heading strong { font-size: 0.88rem; }
    .source-count {
      color: #777f8d; background: #f0f1f3;
      border-radius: 999px; font-size: 0.72rem; padding: 3px 8px;
    }
    .source-card {
      border: 1px solid #e7e9ee;
      background: white;
      border-radius: 12px;
      padding: 11px 12px;
      margin-bottom: 8px;
      box-shadow: 0 3px 12px rgba(23,31,44,.025);
    }
    .source-card:hover { border-color: #cfd3de; }
    .source-index {
      width: 22px; height: 22px; border-radius: 7px;
      background: var(--brand-soft); color: var(--brand);
      display: inline-grid; place-items: center;
      font-size: 0.72rem; font-weight: 800; margin-right: 7px;
    }
    .source-name {
      font-size: 0.78rem; font-weight: 720;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      max-width: 190px; display: inline-block; vertical-align: middle;
    }
    .source-page { color: #888f9c; font-size: 0.72rem; margin-top: 2px; }
    .source-excerpt {
      color: #667085; font-size: 0.76rem; line-height: 1.5;
      margin-top: 8px; display: -webkit-box;
      -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;
    }
    .empty-sources {
      border: 1px dashed #dde1e7; border-radius: 12px;
      padding: 24px 14px; text-align: center;
      color: #8a919d; font-size: 0.78rem;
    }
    .document-row {
      border: 1px solid #e4e6ea; background: white;
      border-radius: 10px; padding: 8px 9px;
      font-size: 0.76rem; margin-bottom: 6px;
    }
    .document-row strong { display: block; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
    .document-row span { color: #9299a5; font-size: 0.68rem; }
    .composer-status { color: #7a8290; font-size: 0.72rem; text-align: center; margin-top: 6px; }
    @media (max-width: 900px) {
      [data-testid="stSidebar"] { min-width: 210px; }
      .source-panel { display: none; }
      .topbar { padding: 0 0.8rem; margin-left: -0.8rem; margin-right: -0.8rem; }
      .welcome { margin-top: 8vh; }
      .prompt-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown('<div class="workspace-title">Knowledge Studio</div>', unsafe_allow_html=True)
    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.document_index = None
        st.session_state.indexed_files = []
        st.rerun()

    st.markdown('<div class="side-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Knowledge base</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Build index", use_container_width=True, disabled=not uploaded):
        try:
            if not uploaded:
                raise ValueError("Upload at least one document")
            config = AppConfig.from_env()
            required_key = "OPENAI_API_KEY" if config.embedding_provider == "openai" else "GEMINI_API_KEY"
            if not os.getenv(required_key):
                raise ValueError("Add your API key in Settings before indexing")
            index = DocumentIndex(build_embeddings(config))
            saved_paths = []
            names = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded:
                    path = os.path.join(temp_dir, file.name)
                    with open(path, "wb") as handle:
                        handle.write(file.getbuffer())
                    saved_paths.append(path)
                    names.append(file.name)
                count = index.load_files(saved_paths)
            st.session_state.document_index = index
            st.session_state.indexed_files = names
            st.success(f"{count} chunks indexed")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.markdown(
                f'<div class="document-row"><strong>▤  {name}</strong><span>Indexed</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed")

    st.markdown('<div class="side-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Tools</div>', unsafe_allow_html=True)
    enable_web = st.toggle("Web search", value=False)

    st.markdown('<div class="side-section"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
    recent = [m["content"] for m in st.session_state.messages if m["role"] == "user"][-5:]
    if recent:
        for item in reversed(recent):
            st.caption(item[:42] + ("…" if len(item) > 42 else ""))
    else:
        st.caption("Your conversations will appear here")


header_left, header_right = st.columns([8.2, 1.8], vertical_alignment="center")
with header_left:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <div class="brand-mark">K</div>
            <div>
              <div class="brand-name">Knowledge Studio</div>
              <div class="brand-status"><span class="status-dot"></span>Grounded workspace</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)
    with st.popover("⚙  Settings", use_container_width=True):
        st.subheader("Configuration")
        provider = st.selectbox(
            "Model provider",
            options=["gemini", "openrouter", "openai"],
            index=["gemini", "openrouter", "openai"].index(
                st.session_state.provider
                if st.session_state.provider in {"gemini", "openrouter", "openai"}
                else "gemini"
            ),
        )
        st.session_state.provider = provider
        os.environ["AI_PROVIDER"] = provider

        if provider == "openai":
            key = st.text_input(
                "OpenAI API key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-...",
            )
            if key:
                os.environ["OPENAI_API_KEY"] = key
            os.environ["EMBEDDING_PROVIDER"] = "openai"
        elif provider == "gemini":
            key = st.text_input(
                "Gemini API key",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                placeholder="Enter API key",
            )
            if key:
                os.environ["GEMINI_API_KEY"] = key
            os.environ["EMBEDDING_PROVIDER"] = "gemini"
        else:
            key = st.text_input(
                "OpenRouter API key",
                type="password",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                placeholder="sk-or-v1-...",
            )
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
            embedding_key = st.text_input(
                "Gemini API key for embeddings",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                placeholder="Enter API key",
            )
            if embedding_key:
                os.environ["GEMINI_API_KEY"] = embedding_key
            os.environ["EMBEDDING_PROVIDER"] = "gemini"
            st.caption("Free chat models: openrouter/free")

        tavily_key = st.text_input(
            "Tavily key",
            type="password",
            value=os.getenv("TAVILY_API_KEY", ""),
            placeholder="Optional for web search",
        )
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key

        if provider == "openai":
            required_key = "OPENAI_API_KEY"
        elif provider == "gemini":
            required_key = "GEMINI_API_KEY"
        else:
            required_key = "OPENROUTER_API_KEY"
        st.caption("✓ API key connected" if os.getenv(required_key) else "Free API key required")


chat_col, source_col = st.columns([8.2, 3.2], gap="large")

with chat_col:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome">
              <div class="welcome-kicker">Research workspace</div>
              <h1>Ask better questions.<br>Get grounded answers.</h1>
              <p>Upload a knowledge source, then explore it with document retrieval and verifiable page citations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt_a, prompt_b = st.columns(2)
        prompt_c, prompt_d = st.columns(2)
        with prompt_a:
            if st.button("Summarize the document", use_container_width=True):
                st.session_state.pending_prompt = "Summarize the document and highlight the main ideas."
                st.rerun()
        with prompt_b:
            if st.button("Find key definitions", use_container_width=True):
                st.session_state.pending_prompt = "Identify and explain the most important definitions."
                st.rerun()
        with prompt_c:
            if st.button("Extract important numbers", use_container_width=True):
                st.session_state.pending_prompt = "Extract important numbers, dates, and figures."
                st.rerun()
        with prompt_d:
            if st.button("What should I know?", use_container_width=True):
                st.session_state.pending_prompt = "What are the most important things I should know?"
                st.rerun()
    else:
        for message in st.session_state.messages:
            with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    c1, c2, c3 = st.columns([0.45, 0.45, 8])
                    with c1:
                        st.button("↻", key=f"retry-{len(st.session_state.messages)}-{message['content'][:12]}", help="Retry")
                    with c2:
                        st.button("☰", key=f"copy-{len(st.session_state.messages)}-{message['content'][:12]}", help="Copy")

    question = st.session_state.pending_prompt
    if question is None:
        question = st.chat_input("Ask about your documents...")
    else:
        st.session_state.pending_prompt = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                config = AppConfig.from_env()
                if config.provider == "openai":
                    required_key = "OPENAI_API_KEY"
                elif config.provider == "gemini":
                    required_key = "GEMINI_API_KEY"
                else:
                    required_key = "OPENROUTER_API_KEY"
                if not os.getenv(required_key):
                    raise ValueError("Open Settings and enter an API key to start.")

                index = st.session_state.document_index
                rag_tool = None
                retrieved = []

                if index is not None:
                    retrieved = index.retrieve(question, k=5)
                    st.session_state.sources = [
                        {
                            "source": chunk.metadata.get("source", "unknown"),
                            "page": (chunk.metadata.get("page") + 1)
                            if chunk.metadata.get("page") is not None
                            else None,
                            "excerpt": chunk.page_content,
                        }
                        for chunk in retrieved
                    ]

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
                        description="Search uploaded documents and return passages with file and page citations.",
                    )

                web_tool = None
                if enable_web:
                    if config.tavily_key:
                        from langchain_community.tools.tavily_search import TavilySearchResults

                        web_tool = TavilySearchResults(
                            max_results=5,
                            tavily_api_key=config.tavily_key,
                        )
                    else:
                        from langchain_community.tools import DuckDuckGoSearchRun

                        web_tool = DuckDuckGoSearchRun(
                            name="web_search",
                            description="Search the live web with DuckDuckGo.",
                        )

                with st.status("Researching your request", expanded=False) as status:
                    if rag_tool is not None:
                        status.update(label="Searching the knowledge base")
                    chat_model = build_chat_model(config)

                    if index is not None and rag_tool is None:
                        context = "\n\n".join(
                            f"[Source: {source['source']}"
                            f"{', page ' + str(source['page']) if source['page'] else ''}] "
                            f"{source['excerpt']}"
                            for source in st.session_state.sources
                        )
                        messages = [
                            SystemMessage(
                                content="Answer strictly from the supplied document context. "
                                "Cite filename and page for factual claims. "
                                "If the context is insufficient, say so clearly."
                            ),
                            HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
                        ]
                        answer = "".join(
                            chunk.content
                            for chunk in chat_model.stream(messages)
                            if isinstance(chunk.content, str)
                        )
                    else:
                        executor = build_agent(chat_model, web_tool, rag_tool)
                        result = executor.invoke(
                            {
                                "input": question,
                                "chat_history": [
                                    {
                                        "role": message["role"],
                                        "content": message["content"],
                                    }
                                    for message in st.session_state.messages[:-1]
                                ],
                            }
                        )
                        answer = result["output"]
                    status.update(label="Answer ready", state="complete")

            except Exception as exc:
                answer = f"Couldn’t complete the request: {exc}"
                st.session_state.sources = []

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

with source_col:
    st.markdown('<div class="source-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="source-heading"><strong>Sources</strong>'
        f'<span class="source-count">{len(st.session_state.sources)}</span></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.sources:
        for number, source in enumerate(st.session_state.sources, start=1):
            page = f"Page {source['page']}" if source["page"] else "Document"
            excerpt = source["excerpt"].replace("\n", " ").strip()
            st.markdown(
                f"""
                <div class="source-card">
                  <div><span class="source-index">{number}</span>
                  <span class="source-name">{source['source']}</span></div>
                  <div class="source-page">{page}</div>
                  <div class="source-excerpt">{excerpt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="empty-sources">Sources will appear here after retrieval.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
