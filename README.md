# LangChain RAG & Tool Agent

A Streamlit application covering internship Tasks 2, 3, and 4:

- PDF/TXT/MD/CSV loading
- Recursive text chunking
- FAISS vector retrieval
- Strict, source-grounded answers with filename/page citations
- OpenAI or Gemini chat and embeddings
- Calculator and local file tools
- Optional Tavily web search
- Streaming-ready chat interface with session history

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

For OpenAI:

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
```

For Gemini:

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_PROVIDER=gemini
```

For web search, optionally set `TAVILY_API_KEY` and enable web search in the sidebar.

## Run

```powershell
streamlit run app.py
```

Upload a PDF or text document, build the document index, and ask a question. Answers from documents include source filenames and PDF page numbers. If the documents do not contain the answer, the app is prompted to state that rather than inventing a response.

## Tests

```powershell
python -m pytest
```

## Safety

API keys remain in environment variables and are never committed. The local file tool is restricted to the project workspace and has a 200 KB size limit.
