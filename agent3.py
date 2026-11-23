from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode

load_dotenv()



class TenderAnalysis(BaseModel):
    detected_language : str = Field(description="detected language of the input text")
    sector : str = Field(description="detected sector of the input text")
    budget_estimate : str = Field(description="estimated budget of the input text")
    stakeholders : list[str] = Field(description="list of stakeholders mentioned in the input text")
    action_items : list[str] = Field(description="list of action items mentioned in the input text")
    relevance_score : float = Field(description="relevance score of the input text to the project ge=0.0, le=1.0")
    executive_summary : str = Field(description="executive summary of the input text")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_output: Annotated[TenderAnalysis, add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

structured_llm = llm.with_structured_output(TenderAnalysis)

# with this pydantic the model returns the instance of the TenderAnalysis class.


def get_tender_analysis(state:AgentState) -> AgentState:
    """ Get a structured analysis of a tender document."""
    response = structured_llm.invoke(state["messages"])
    return { "structured_output" : [response]}



builder = StateGraph(AgentState)
builder.add_node("get_tender_analysis", get_tender_analysis)

builder.add_edge(START, "get_tender_analysis")
builder.add_edge("get_tender_analysis", END)

memory = MemorySaver()

app = builder.compile(checkpointer=memory)

config = {"configurable":{"thread_id":"terminal-session-1"}}


policy_brief = """
AVIS D'APPEL D'OFFRES OUVERT N° 14/2026/MTNRA
Dans le cadre de la stratégie Maroc Digital 2030, le Ministère de la Transition Numérique et de la Réforme de l'Administration (MTNRA), en partenariat avec l'Agence de Développement du Digital (ADD), lance un projet pilote visant le déploiement d'un assistant intelligent basé sur l'IA générative pour moderniser l'accueil et l'orientation des citoyens au sein des centres de services publics de Rabat et Casablanca. 
Le budget prévisionnel alloué à cette prestation s'élève à environ 450 000 USD (soit près de 4,5 millions de dirhams), cofinancé par la Banque Mondiale.
Le prestataire retenu devra : (1) concevoir l'architecture RAG multilingue (Darija, Arabe standard, Français), (2) intégrer les APIs avec le portail national Idarati.ma, et (3) former les équipes techniques de l'ADD avant le déploiement final prévu au T3 2026. Les citoyens usagers des services administratifs seront les bénéficiaires directs de cette solution.
"""

response = app.invoke(
    {"messages":[HumanMessage(content=policy_brief)]},
    config=config
)

print(response["structured_output"])











