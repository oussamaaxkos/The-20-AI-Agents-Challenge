from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class APIReturn(BaseModel):
    status_code: int = Field(description="HTTP status code returned by the API.")
    response_body: str = Field(description="The body of the response returned by the API.")

llm_with_structured_output = llm.with_structured_output(APIReturn)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    api_response: Optional[APIReturn]  # Separate field for the Pydantic object
    retries: int
    max_retries: int

def get_api_response(state: AgentState) -> dict:
    """Get a structured API response from the LLM."""
    response = llm_with_structured_output.invoke(state["messages"])
    return {
        "api_response": response,
        "messages": [AIMessage(content=response.model_dump_json())]  # Must be an AIMessage
    }

def validate_api_call(state: AgentState) -> dict:
    """Validate status_code and update retries + error feedback in state."""
    api_response = state["api_response"]
    if api_response and api_response.status_code == 200:
        return {"retries": 0}

    new_retries = state["retries"] + 1
    status = api_response.status_code if api_response else "None"
    return {
        "retries": new_retries,
        "messages": [HumanMessage(content=f"API call failed with status code: {status}. Fix it and return status 200.")]
    }

def conditionally_call_api(state: AgentState) -> str:
    """Pure routing function based on updated state."""
    if state["api_response"] and state["api_response"].status_code == 200:
        return "end"
    if state["retries"] >= state["max_retries"]:
        return "max_retries_reached"
    return "retrie"

def fallback_node(state: AgentState) -> dict:
    """Fallback node when maximum retries have been reached."""
    return {"messages": [AIMessage(content="Maximum number of retries reached. Please try again later.")]}

builder = StateGraph(AgentState)
builder.add_node("get_api_response", get_api_response)
builder.add_node("validate_api_call", validate_api_call)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "get_api_response")
builder.add_edge("get_api_response", "validate_api_call")
builder.add_conditional_edges(
    "validate_api_call",
    conditionally_call_api,
    {
        "retrie": "get_api_response",
        "max_retries_reached": "fallback",
        "end": END
    }
)
builder.add_edge("fallback", END)

memory = MemorySaver()
app = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "terminal-session-1"}}

response = app.invoke(
    {
        "messages": [HumanMessage(content="Simulate a failed API response with status 500 first, then fix it on retry.")],
        "api_response": None,
        "retries": 0,
        "max_retries": 3
    },
    config=config
)

print("Response from the LLM:", response["messages"][-1].content)
