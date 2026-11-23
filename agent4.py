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
    retries: int = 0
    max_retries: int = 3

def get_api_response(state:AgentState) -> AgentState:
    """ Get a structured analysis of a tender docuement. """
    response = llm_with_structured_output.invoke(state["messages"])
    return {"messages":[response]}


def conditionally_call_api(state:AgentState) -> AgentState:
    """ catch the response from the LLM and call the API if the status code is 200. """
    
    try:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            api_response = last_message.content
            if api_response.status_code == 200:
                # Call the API with the response body
                # Here you would implement the actual API call using a library like requests
                state["retries"] = 0
                return "end"  # Indicate that the process is complete
            else:
                raise ValueError(f"API call failed with status code: {api_response.status_code}")
    except ValueError as e:
        state["retries"] += 1
        if state["retries"] == state["max_retries"]:
            return "max_retries_reached"  # Indicate that the maximum number of retries has been reached
        return "retrie"


