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

from .components.state import RagStateInput, RagState, RagStateOutput_naive
from .components.configuration import Configuration
from .components.llm_handlers import augment_context, generate_answers
from .components.retriever import create_retriever

ROOT = pathlib.Path("/workspace")

# retrieval
def retrieve_raw_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    question = state["question"]

    # create/load the retriever
    retriever = create_retriever(
        top_k=cfg.top_k,
        subfolder=ROOT / "data/vectorstores",
        )

    # plain vector retrieval (no compression, no rerank)                      
    docs = retriever.invoke(question)

    return {
        "context": docs,
    }

# augmentation
def augment_step(state: RagState, config: RunnableConfig):
    """Augment the system prompt with retrieved context."""
    question = state["question"]
    context = state["context"]
    
    augmented_prompt = augment_context(
        question=question,
        context_docs=context
    )
    
    return {
        "augmented_prompt": augmented_prompt
    }

# generation
def generate_step(state: RagState, config: RunnableConfig):
    """Generate answers using the augmented prompt."""
    configurable = Configuration.from_runnable_config(config)
    question = state["question"]
    augmented_prompt = state["augmented_prompt"]
    
    answers = generate_answers(
        question=question,
        augmented_prompt=augmented_prompt,
        cfg=configurable
    )
    
    return {
        "answers": answers
    }
        
def create_rag_graph():
    builder = StateGraph(
        RagState,
        input = RagStateInput,
        output = RagStateOutput_naive,
        config_schema = Configuration,
    )

    # Simple setup: Retrieve → Generate
    builder.add_node("retrieve", retrieve_raw_step)
    builder.add_node("augment", augment_step)
    builder.add_node("generate", generate_step)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "augment")
    builder.add_edge("augment", "generate")
    builder.add_edge("generate", END)

    return builder.compile(name="rag-agent-simple")

# Create the graph instance
graph = create_rag_graph()