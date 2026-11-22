from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import math



load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]






llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


@tool
def addition(numbers: list[float]) -> float:
    """Add multiple numbers together (e.g., [2, 5, 10] -> 17)."""
    return sum(numbers)


@tool
def subtraction(numbers: list[float]) -> float | str:
    """Subtract all subsequent numbers from the first number (e.g., [100, 20, 5] -> 75)."""
    if not numbers:
        return "Error: Provide at least one number."
    result = numbers[0]
    for n in numbers[1:]:
        result -= n
    return result


@tool
def multiply(numbers: list[float]) -> float | str:
    """Multiply multiple numbers together (e.g., [2, 3, 4] -> 24)."""
    if not numbers:
        return "Error: Provide at least one number."
    return math.prod(numbers)


@tool
def division(numbers: list[float]) -> float | str:
    """Divide the first number sequentially by all subsequent numbers (e.g., [100, 2, 5] -> 10)."""
    if len(numbers) < 2:
        return "Error: Provide at least two numbers for division."
    result = numbers[0]
    for n in numbers[1:]:
        if n == 0:
            return "Error: Division by zero is undefined."
        result /= n
    return result


@tool
def average(numbers: list[float]) -> float | str:
    """Calculate the arithmetic mean (average) of multiple numbers."""
    if not numbers:
        return "Error: Provide at least one number."
    return sum(numbers) / len(numbers)


@tool
def min_max(numbers: list[float]) -> dict[str, float] | str:
    """Find the minimum and maximum values from a list of numbers."""
    if not numbers:
        return "Error: Provide at least one number."
    return {"min": min(numbers), "max": max(numbers)}


@tool
def power(base: float, exponent: float) -> float:
    """Raise a base number to an exponent (base ^ exponent)."""
    return base**exponent


@tool
def square_root(x: float) -> float | str:
    """Calculate the square root of a non-negative number."""
    if x < 0:
        return "Error: Cannot calculate the square root of a negative number."
    return math.sqrt(x)


@tool
def modulo(a: float, b: float) -> float | str:
    """Find the remainder of dividing a by b (a % b)."""
    if b == 0:
        return "Error: Modulo by zero is undefined."
    return a % b


@tool
def percentage(value: float, percent: float) -> float:
    """Calculate a percentage of a given value (e.g., 20% of 150 -> value=150, percent=20)."""
    return (value * percent) / 100


@tool
def factorial(n: int) -> int | str:
    """Calculate the factorial of a non-negative integer (n!)."""
    if n < 0:
        return "Error: Factorial is not defined for negative numbers."
    return math.factorial(n)


tools = [
    addition,
    subtraction,
    multiply,
    division,
    average,
    min_max,
    power,
    square_root,
    modulo,
    percentage,
    factorial,
]


llm_with_tools  = llm.bind_tools(tools)


def chat(state:AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

def condition_tools(state:AgentState):
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls :
        return "tools"
    return END

tool_node = ToolNode(tools)


builder = StateGraph(AgentState)

builder.add_node("chat", chat)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chat")
builder.add_conditional_edges("chat",
      condition_tools,
      {
          "tools":"tools",
          END:"chat"
      }
      
      )

builder.add_edge("tools","chat")



memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "terminal-session-1"}}

# 3. Terminal chat loop
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    print(f"Agent: {response['messages'][-1].content}\n")