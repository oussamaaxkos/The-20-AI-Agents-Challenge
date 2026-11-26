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
import math

load_dotenv()

    












