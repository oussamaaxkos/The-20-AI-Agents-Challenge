from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import math


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
)

class CountryBudgetResult(BaseModel):
    budget: float = Field(description=("The country budget in digital transformation."))


structured_llm = llm.with_structured_output(CountryBudgetResult)


@tool
def lookup_country_budget(country_name:str) -> float :
    """ Get the morocco budget in digital transformation """
    response = structured_llm.invoke(f"Get the country budget in digital transformation for {country_name} and return only the budget.")
    return response.budget

@tool
def Calculate_deference(ctr1:float, ctr2:float) -> float:
    """
    IMPORTANT: Do NOT call this tool until you have already retrieved the exact 
    numeric budgets for both countries using `lookup_country_budget`. Never guess values.
    """
    response = abs(ctr1 - ctr2)
    return response

tools = [lookup_country_budget, Calculate_deference]

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a ReAct agent.
Given this conversation so far and this JSON menu of tools, decide the next logical step:
- If you are missing required information, write a short 'Thought:' explaining what is missing and call the appropriate tool.
- Only give your final answer once all required observations are complete."""

def thought(state:AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT) + state["messages"]]
    response = llm_with_tools.invoke(messages)
    # here the llm can see the message and tools, and decide which tool to call next, or to respond directly to the user
    #  {
    #   "name": "lookup_country_budget",
    #   "description": "Get a given country's budget in digital transformation in USD.",
    #   "parameters": {
    #     "type": "object",
    #     "properties": {
    #       "country_name": { "type": "string" }
    #     },
    #     "required": ["country_name"]
    #   }
    # }
    return {"messages":[response]}

def decision(state:StateGraph):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"

act_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("tought", thought)
builder.add_node("act_node", act_node)
builder.add_node("decision", decision)

builder.add_edge(START, "tought")
builder.add_conditional_edges(
                "tought",
                 decision,
                 {
                    "tools":"act_node",
                    "end":END 
                 }
)

builder.add_edge("act_node", "tought")

memory_saver = MemorySaver()

app = builder.compile(checkpointer=memory_saver)
config = {"configurable":{"thread_id":"terminal-session-1"}}

response = app.invoke(
    {
        "message":[HumanMessage(content="Compare the 2026 digital transformation budget of Morocco with Tunisia's, and calculate the percentage difference.")]
    },
    config=config
)










