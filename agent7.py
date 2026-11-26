# 07: Reflection agent and Self-Correction Agent introduces
# an iteraive quality-control loop where an LLM generates a draft
# critiques its own work against specific rules
# and revises the output until it passes or hits a maximum retry limit.

# why Agnet 07 Exists (Singke-Pass vs Self-Critique)
# In earlier agents, once the LLM produces a final answer
# that text goes straight to the user
# even if it missed a constraint, hallucinated an edge case, or produced a buggy code.

# Agent 07 separates Generation from critique
# how a developer writes code, runs a code review, fixes bugs, and re-tests
# before shipping.

# Graph Topology (Generator-Critic Loop)
# Start -> Generator Node -> Critic NODE (Evaluates draft against strict rules) (Outputs structured Critique (Pydatntic)) -> GENERATOR NODE (Rewrites draft based on Critique) -> Critic Node (Evaluates new draft) -> Generator Node (Rewrites again) -> ... -> END
# [passed == True OR max_revisions] -> END

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
)



class AgentState(TypedDict):
     messages : Annotated[list[AnyMessage], add_messages]
     query : str
     is_critic : bool
     max_critic : int


class IsCritic(BaseModel):
     is_critic : bool = Field(description = "Is this answer write compared with this query")

class CreateStructuredQuery(BaseModel):
     query : str = Field(description = "use this query and generate a structured one.")

is_critic_point_llm = llm.with_structured_output(IsCritic)
structured_query = llm.with_structured_output(CreateStructuredQuery)




@tool
def information_retrieval(query:str):
    chunks = ["a","b","c"]
    return chunks


tools = [information_retrieval]


step_executor_agent = create_agent(model=llm, tools=tools)

def GenertorNode(state:AgentState):
    if state['is_critic'] :
        state["query"] = structured_query.invoke(f"Here is the context reponse and critics points:{state['messages']} update query based on crtic points")
    response = step_executor_agent.invoke(state["query"])
    return {"messages":response}

def CriticNode(state:AgentState):
    is_critic = is_critic_point_llm.invoke(f"query : {state['query']}, response : {state['messages'][-1]}")
    critic_point = None
    if is_critic :
        critic_point = llm.invoke(f"here is the query {state['query']} and here is the answer {state['messages']} what is the critic point")
        state["max_critic"] -= 1
    return {"messages":critic_point, "is_critic":is_critic}

def condtional_edge(state:AgentState):
    if state["max_critic"] == -1:
        return "end"
    return "generator" if state["is_critic"] else "end"


builder = StateGraph(AgentState)
app = builder.compile()

app.add_node("generator", GenertorNode)
app.add_node("critic", CriticNode)

app.add_edge(START, "generator")
app.add_edge("generator", "critic")

app.add_conditional_edges(
    "critic",
    condtional_edge,
    {
        "generator": "generator",
        "end": END
    }
)





