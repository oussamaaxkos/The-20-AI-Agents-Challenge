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

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

class APIReturn(BaseModel):
    status_code: int = Field(description="HTTP status code returned by the API.")
    response_body: str = Field(description="The body of the response returned by the API.")


llm_with_structured_output = llm.with_structured_output(APIReturn)




class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]

def get_api_response(state:AgentState) -> AgentState:
    """ Get a structured analysis of a tender docuement. """
    response = llm_with_structured_output.invoke(state["messages"])
    return {"messages":[response]}


def conditionally_call_api(state:AgentState) -> AgentState:
    """  """


