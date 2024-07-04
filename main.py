import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from docx import Document
import graphviz
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Print the SAM API key for debugging
print("SAM_API_KEY:", os.getenv("SAM_API_KEY"))

def fetch_sam_opportunities(limit=5):
    url = "https://api.sam.gov/prod/opportunities/v2/search"
    params = {
        "limit": limit,
        "postedFrom": "01/01/2021",
        "postedTo": "12/25/2021",
        "api_key": os.getenv("SAM_API_KEY")
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {"error": response.json()}

@app.get("/sam_opportunities")
def get_sam_opportunities(limit: int = 5):
    return fetch_sam_opportunities(limit)

# Define the AgentState type
class AgentState(TypedDict):
    task: str
    plan: str
    research_plan_output: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# Initialize the OpenAI model
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.0")

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content="Please provide a plan for the following task."),
        HumanMessage(content=state['task'])
    ]
    response = model.predict_messages(messages)
    state.update({"plan": response.content})
    return state

def research_plan_node(state: AgentState):
    messages = [
        SystemMessage(content="Based on the plan, provide research steps."),
        HumanMessage(content=f"Plan: {state['plan']}")
    ]
    response = model.predict_messages(messages)
    state.update({"research_plan_output": response.content})
    return state

def generation_node(state: AgentState):
    messages = [
        SystemMessage(content="Generate content based on the plan and research."),
        HumanMessage(content=f"Plan: {state['plan']}\nResearch: {state['research_plan_output']}")
    ]
    response = model.predict_messages(messages)
    state.update({"draft": response.content})
    return state

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content="Reflect on the generated content and provide feedback."),
        HumanMessage(content=f"Draft: {state['draft']}")
    ]
    response = model.predict_messages(messages)
    state.update({"critique": response.content})
    return state

def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content="Provide a final critique and suggestions for improvement."),
        HumanMessage(content=f"Draft: {state['draft']}\nCritique: {state['critique']}")
    ]
    response = model.predict_messages(messages)
    state.update({"critique": response.content, "revision_number": state["revision_number"] + 1})
    return state

def end_node(state: AgentState):
    return state

# Build and compile the state graph
builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)
builder.add_node("end", end_node)

builder.set_entry_point("planner")

# Conditional edge to handle revision logic
builder.add_conditional_edges(
    "reflect",
    lambda state: state["revision_number"] < state["max_revisions"],
    {True: "research_critique", False: "end"}
)

# Defining the flow of the graph
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("generate", "reflect")
builder.add_edge("research_critique", "generate")

graph = builder.compile()

@app.post("/run_graph")
def run_graph(task: str, max_revisions: int = 3):
    initial_state = AgentState(
        task=task,
        plan="",
        research_plan_output="",
        draft="",
        critique="",
        content=[],
        revision_number=0,
        max_revisions=max_revisions
    )

    config = {
        "configurable": {
            "thread_id": "example_thread_id",
            "thread_ts": "example_thread_ts"
        }
    }

    for step, output in enumerate(graph.stream(initial_state, config=config)):
        if output.get("lnode") == "end":
            break

    return JSONResponse(content=output)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
