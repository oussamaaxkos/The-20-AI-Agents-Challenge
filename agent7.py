from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    draft: str
    needs_revision: bool
    max_critic: int


class CritiqueResult(BaseModel):
    needs_revision: bool = Field(
        description="True if the answer is incomplete, inaccurate, or misses part of the query; False if it is fully accurate."
    )
    feedback: str = Field(
        description="Specific instructions on what to fix or retrieve next, or 'Approved' if needs_revision is False."
    )


class RewrittenQuery(BaseModel):
    query: str = Field(
        description="An improved, targeted search/generation query incorporating the critic's feedback."
    )


critic_llm = llm.with_structured_output(CritiqueResult)
query_rewriter_llm = llm.with_structured_output(RewrittenQuery)


@tool
def information_retrieval(query: str) -> list[str]:
    """Retrieve relevant knowledge chunks for a given query."""
    chunks = [
        "Chunk A: Morocco's 2026 digital strategy allocates $450M to AI and cloud infrastructure.",
        "Chunk B: Tunisia's 2026 digital plan focuses on startup ecosystems with a $300M budget.",
        "Chunk C: Regional fiber-optic connectivity between North Africa and Europe expanded by 35%.",
    ]
    return chunks


tools = [information_retrieval]
step_executor_agent = create_agent(model=llm, tools=tools)


def generator_node(state: AgentState) -> dict:
    current_query = state["query"]

    if state.get("needs_revision"):
        rewritten: RewrittenQuery = query_rewriter_llm.invoke(
            f"Original query: {current_query}\n"
            f"Previous draft: {state['draft']}\n"
            f"Critic feedback: {state['messages'][-1].content}\n"
            f"Rewrite the query to resolve the critic's feedback."
        )
        current_query = rewritten.query

    result = step_executor_agent.invoke({"messages": [HumanMessage(content=current_query)]})
    latest_answer = result["messages"][-1]

    return {
        "query": current_query,
        "draft": latest_answer.content,
        "messages": [latest_answer],
    }


def critic_node(state: AgentState) -> dict:
    evaluation: CritiqueResult = critic_llm.invoke(
        f"User Query: {state['query']}\n"
        f"Generated Answer: {state['draft']}\n"
        f"Evaluate whether the answer completely and accurately addresses the query."
    )

    updates: dict = {
        "needs_revision": evaluation.needs_revision,
        "max_critic": state["max_critic"] - 1,
    }

    if evaluation.needs_revision:
        updates["messages"] = [HumanMessage(content=f"Critic Feedback: {evaluation.feedback}")]

    return updates


def conditional_edge(state: AgentState) -> str:
    if not state["needs_revision"] or state["max_critic"] <= 0:
        return "end"
    return "generator"


builder = StateGraph(AgentState)
builder.add_node("generator", generator_node)
builder.add_node("critic", critic_node)

builder.add_edge(START, "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges(
    "critic",
    conditional_edge,
    {
        "generator": "generator",
        "end": END,
    },
)

memory_saver = MemorySaver()
app = builder.compile(checkpointer=memory_saver)


config = {"configurable": {"thread_id": "terminal-session-2"}}
user_query = "Summarize relevant details from chunks a and b"

response = app.invoke(
    {
        "messages": [HumanMessage(content=user_query)],
        "query": user_query,
        "draft": "",
        "needs_revision": False,
        "max_critic": 2,
    },
    config=config,
)

print("Final RAG Answer:\n", response["draft"])