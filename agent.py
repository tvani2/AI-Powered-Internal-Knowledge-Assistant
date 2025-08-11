import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate

from tools.database_tool import DatabaseTool
from tools.document_search_tool import DocumentSearchTool


SYSTEM_PREFIX = (
    "You are a helpful enterprise assistant. Always use tools accurately. "
    "For database questions, first call sql_schema to inspect tables/columns, then compose a single valid SQLite SELECT for sql_database. "
    "Never attempt multi-statement SQL. Prefer concise, factual answers with citations/snippets when using document_search."
)


def build_agent() -> AgentExecutor:
    load_dotenv()

    # Initialize tools
    db_tool = DatabaseTool()
    doc_tool = DocumentSearchTool()

    tools = [
        db_tool.as_schema_tool(),
        db_tool.as_langchain_tool(),
        doc_tool.as_langchain_tool(),
    ]

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    # Use OpenAI functions-style agent to encourage structured tool calls
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": SYSTEM_PREFIX,
        },
    )
    return agent


def run(query: str) -> str:
    agent = build_agent()
    return agent.run(query)


if __name__ == "__main__":
    print(run("List top 5 employees by total sales revenue. Show name and total."))
