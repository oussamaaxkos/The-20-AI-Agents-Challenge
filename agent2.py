from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import math

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class CalculationResult(BaseModel):
    result: int = Field(description="The result of the calculation.")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

structured_llm = llm.with_structured_output(CalculationResult)

@tool
def calculate(args: str) -> str:
    """Perform a calculation based on the provided arguments."""
    response = structured_llm.invoke(f"Calculate the following and return only the integer: {args}")
    return response.result

@tool
def get_weather(location:str) -> str:
    """Get the current weather for a given location."""
    response = llm.invoke(f"Get the current weather for {location} and return only the weather description.")
    return response.result

@tool
def get_news(topic:str) -> str:
    """Get the latest news for a given topic."""
    response = llm.invoke(f"Get the latest news for {topic} and return only the news headline.")
    return response.result


@tool
def get_organization_info(organization_name:str) -> str:
    """Get information about a given organization."""
    response = llm.invoke(f"Get information about the organization {organization_name} and return only the description.")
    return response.result

tools = [get_weather, get_news, get_organization_info, calculate]

llm_with_tools = llm.bind_tools(tools)

def chat(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def conditional_edge(state:AgentState) -> bool:
    cond = state["messages"][-1].tool_calls
    if cond:
        return "tools"
    return END
    


tools_node = ToolNode(tools)

builder = StateGraph(AgentState)

builder.add_node("chat",chat)
builder.add_node("tools_node", tools_node)

builder.add_edge(START,"chat")
builder.add_conditional_edges(
    "chat",
    conditional_edge,
    {
        "tools":"tools_node",
        END:END
    }
)

builder.add_edge("tools_node","chat")

memory = MemorySaver()

app = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "terminal-session-1"}}

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print(f"Agent: {response['messages'][-1].content}\n")


