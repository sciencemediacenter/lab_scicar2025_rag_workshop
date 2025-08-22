# graph_agent/graph.py
import os
from dotenv import load_dotenv
import langchain

import re
from typing import List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

import pandas as pd
import pathlib

from .components.state import RagStateInput, RagState, RagStateOutput
from .components.configuration import Configuration
from .components.llm_handlers import handle_anthropic_generation
from .components.retriever import create_retriever

ROOT = pathlib.Path("/workspace")

# create/load the retriever
retriever = create_retriever(
    top_k=10,
    subfolder=ROOT / "data/vectorstores",
    )

# retrieval
def retrieve_raw_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    question = state["question"]

    # plain vector retrieval (no compression, no rerank)                      
    docs = retriever.invoke(question)

    return {
        "context": docs,
    }

# answer generation
def generate_step(state: RagState, config: RunnableConfig):
    """Generate an answer based on the retrieved context."""
    configurable = Configuration.from_runnable_config(config)
    question = state["question"]
    context = state["context"]

    answers = handle_anthropic_generation(
        question=question, 
        context_docs=context, 
        cfg=configurable
        )
    
    return {
        "answers": answers
    }
        
def create_rag_graph():
    builder = StateGraph(
        RagState,
        input = RagStateInput,
        output = RagStateOutput,
        config_schema = Configuration,
    )

    # Simple setup: Retrieve → Generate
    builder.add_node("retrieve", retrieve_raw_step)
    builder.add_node("get_answer", generate_step)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "get_answer")
    builder.add_edge("get_answer", END)

    return builder.compile(name="rag-agent-simple")

# Create the graph instance
graph = create_rag_graph()