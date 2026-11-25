# [ ] **Agent 06:** Planning & Task Decomposition Agent

# in agent 5 no map 
# imagine driving across Mororcco without GPS - you only look at the next intersection:
# - Short-sighted : (6+ steps), it forgets parts
# - no progress tracking: Everything is mixed together inside one long messages chat history -- you can't easily see 'we are on Step 3 of 5'
# - Expensive : Every single small step re-reads the entire messy conversation history

# How agent 6 solves it (GPS + Checklist)
# Agent 06 splits the brain into two separate jobs:
# 1. The Planner (Writes the Checklist):
# Before calling any tool, it reads your goal and outputs a clean list of steps:
#     1. Get Morocco budget
#     2. Get Tunisia budget
#     3. Calculate difference
#     4. Write final summary

# The Executor (Does One Item at a Time):
# it grabs Step 1, runs the tool, saves the result in past_steps, and crosses Step 1 off the list. Then it moves to Step2

# The Replanner (Updates the List):
# After each step, it checks the remaining list. if Step 1 revealed something unexpected, it edits the remaining checklist; if the checklist is empty, it finishes.

# One-Line Summary
# - Agent 05: Think -> Act -> Think -> Act
# - Agent 06: Make a Ckecklist -> Execute item1 -> Update Checklist -> Execute item2 -> Update Checklist -> Finish

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
    past_response: Optional[str]





