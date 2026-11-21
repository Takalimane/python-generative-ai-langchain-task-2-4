# LangChain RAG & Tool Agent

A Streamlit application covering internship Tasks 2, 3, and 4:

- PDF/TXT/MD/CSV loading
- Recursive text chunking
- FAISS vector retrieval
- Strict, source-grounded answers with filename/page citations
- Gemini or OpenRouter free chat, plus Gemini/OpenAI embeddings
- Calculator and local file tools
- Keyless DuckDuckGo web search, or optional Tavily
- Streaming-ready chat interface with session history

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

### No-payment option: Gemini

Create a Gemini API key in Google AI Studio. It can be used with the API free tier; no
billing or credit card is required for eligible free-tier usage.

In Settings, select **Gemini**, enter the key, and use the app. The key is also used for
free Gemini embeddings.

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_PROVIDER=gemini
```

### No-payment option: OpenRouter

OpenRouter also provides a free-model router. Create an OpenRouter API key and select
**OpenRouter** in Settings. For document embeddings, also enter a Gemini API key.

```dotenv
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openrouter/free
GEMINI_API_KEY=your_gemini_key
EMBEDDING_PROVIDER=gemini
```

### Optional: OpenAI

```dotenv
AI_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=openai
```

Enable web search in the sidebar. It uses DuckDuckGo without an API key by default.
Set `TAVILY_API_KEY` to use Tavily instead.

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
