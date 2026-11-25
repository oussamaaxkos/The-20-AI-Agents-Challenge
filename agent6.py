import operator
from typing import TypedDict, Annotated, Optional, Union
from pydantic import BaseModel, Field

class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of atomic, self-contained steps to achieve the objective.")

class Response(BaseModel):
    response: str = Field(description="The final synthesized answer to the user's objective.")

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Return Response if all steps are complete and you can answer the user; otherwise return Plan with the remaining steps."
    )

class AgentState(TypedDict):
    objective:str
    plan:list[str]
    past_steps : Annotated[list[tuple[str,str]], operator.add]
    final_response: Optional[str]


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

@tool
def lookup_country_budget(country_name:str) -> float:
    """ Get a given country's 2026 digital transformation budget in USD. """
    budgets = {"morocco": 450_000_000.0, "tunisia": 300_000_000.0}
    return budgets.get(country_name.lower(), 0.0)

@tool
def calculate_percentage_difference(val1:float, val2:float) -> float:
    """ Calculate the percentage diffrence between two numeric values.  """
    if val1 == 0 and val2 == 0:
        return 0.0
    try:
        return abs(val1 - val2) / ((val1 + val2) / 2) * 100
    except ZeroDivisionError:
        return float('inf')  # Handle division by zero if both values are zero


tools = [lookup_country_budget, calculate_percentage_difference]

step_executor_agent = create_agent(model=llm,tools=tools)

planner_llm = llm.with_structured_output(Plan)

def planner_node(state:AgentState) -> dict:
    prompt = f""" break down the following objective into a clear, ordered list of small, sekf-contained steps.
    Each step must be executable using a tool or reasoning. Do not include unnecessary fluff.

    Objective: { state['objective'] }
    """
    plan_result: Plan = planner_llm.invoke([HumanMessage(content=prompt)])
    return {"plan": plan_result.steps}

def executor_node(state : AgentState) -> dict:
    current_step = state["plan"][0]

    past_context = "\n".join([f"- {step}:{res}" for step, res in state["past_steps"]])

    task_prompt = f"""Completed facts so far:
    {past_context if past_context else "None"}

    Execute ONLY this single instruction and nothing else:
    {current_step}"""

    result = step_executor_agent.invoke({"messages": [HumanMessage(content=task_prompt)]})
    step_output = result["messages"][-1].content

    return {"past_steps": [(current_step, step_output)]}

replanner_llm = llm.with_structured_output(Act)

def replanner_node(state:AgentState) -> dict:
    past_context = "\n".join([f"- {step}:{res}" for step, res in state["past_steps"]])
    remaining_plan = "\n".join([f"- {step}" for step in state["plan"][1:]])

    prompt = f""" You are a Replanner managing a multi-step task. Original Objective {state['objective']}
    Completed Steps and Results:
    {past_context}
    Remaining Planned Steps:
    {remaining_plan if remaining_plan else "None"}
    Decide what to do next:
    - If you now have all information needed to fully answer the Original Objective, return a 'Response'.
    - Otherwise, return an updated 'Plan' containing ONLY the remaining steps still needed (do NOT repeat completed steps).
    """
    output:Act = replanner_llm.invoke([HumanMessage(content=prompt)])
    if isinstance(output.action, Response):
        return {"final_response":output.action.response}
    else:
        return {"plan":output.action.steps}
    

from langgraph.graph import StateGraph, START, END

def should_end(state: AgentState) -> str:
    if state.get("final_response"):
        return "end"
    return "continue"

builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("replanner", replanner_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor","replanner")
builder.add_conditional_edges(
    "replanner",
    should_end,
    {
        "continue": "executor",
        "end": END
    }
)

app = builder.compile()
inputs = {
    "objective": "Compare the 2026 digital transformation budget of Morocco with Tunisia's, and calculate the percentage difference.",
    "plan": [],
    "past_steps": [],
    "final_response": None
}

for event in app.stream(inputs, stream_mode="updates"):
    for node_name, output in event.items():
        print(f"\n{'='*50}")
        print(f"NODE: {node_name.upper()}")
        print(f"{'='*50}")

        if "plan" in output:
            print("Updated Plan:")
            for i, step in enumerate(output["plan"], 1):
                print(f"  {i}. {step}")

        if "past_steps" in output:
            step, res = output["past_steps"][-1]
            print(f"Completed Step: {step}")
            print(f"Result: {res}")

        if "final_response" in output and output["final_response"]:
            print(f"\nFINAL ANSWER:\n{output['final_response']}")