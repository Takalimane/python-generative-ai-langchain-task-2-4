from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from .tools import build_core_tools


@tool
def grounded_context_search(query: str) -> str:
    """Search uploaded documents for context relevant to the question."""
    return "Upload and index documents before using this tool."


def build_agent(chat_model, web_search=None, rag_tool=None):
    tools = build_core_tools()
    if web_search is not None:
        tools.append(web_search)
    if rag_tool is not None:
        tools.append(rag_tool)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful AI assistant. Choose tools when they help. "
                "For uploaded-document questions, rely on document search and clearly cite filename and page. "
                "If document search is unavailable or its passages do not contain the answer, say that the "
                "uploaded documents do not contain enough information. Never invent a document-based answer. "
                "For real-time questions, use web search. Show the final answer plainly and briefly.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )
