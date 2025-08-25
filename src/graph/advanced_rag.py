# graph_agent/graph.py
import os
from dotenv import load_dotenv
from langsmith import Client
import langchain

import re
from typing import List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from langsmith import traceable
import pandas as pd
import pathlib

from .components.state import RagStateInput, RagState, RagStateOutput_advanced
from .components.configuration import Configuration
from .components.llm_handlers import generate_answers, augment_context
from .components.retriever import create_retriever
from .components.utils import weighted_reciprocal_rank_fusion

ROOT = pathlib.Path("/workspace")

# load prompts for query decomposition
from data.prompts.rag_prompts import (
    DECOMP_PROMPT, 
    CLASSIFY_DECOMP_PROMPT,
    )
    
# query decomposition
def query_decomposition_step(state: RagState, config: RunnableConfig):
    """Determine if query needs decomposition and generate sub-queries if needed."""
    configurable = Configuration.from_runnable_config(config)
    question = state["question"]
    
    if not configurable.use_query_decomposition:
        return {
            "needs_decomposition": False,
            "sub_queries": []
        }
    
    # Set up decision model
    decision_model = ChatOpenAI(
        model=configurable.decomposition_dec_model,
        temperature=0,
        openai_api_key=configurable.openai_api_key
    )
    
    # Check if decomposition is needed
    classification_prompt = CLASSIFY_DECOMP_PROMPT.format(
        query=question.replace('"', '\\"')
    )
    
    try:
        response = decision_model.invoke([HumanMessage(content=classification_prompt)])
        response_text = response.content.strip()
        
        decision_match = re.search(r'<decision>(.*?)</decision>', response_text, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip().upper()
            needs_decomposition = decision == "JA"
        else:
            needs_decomposition = "JA" in response_text.upper() and "NEIN" not in response_text.upper()
        
        if not needs_decomposition:
            return {
                "needs_decomposition": False,
                "sub_queries": []
            }
        
        # Generate sub-queries
        generation_model = ChatOpenAI(
            model=configurable.decomposition_gen_model,
            temperature=0,
            openai_api_key=configurable.openai_api_key
        )
        
        decomposition_prompt = DECOMP_PROMPT.format(
            max_sub_queries=configurable.max_sub_queries,
            query=question.replace('"', '\\"')
        )
        
        response = generation_model.invoke([HumanMessage(content=decomposition_prompt)])
        response_text = response.content.strip()
        
        block = re.search(r"<sub_queries>(.*?)</sub_queries>", response_text, re.S)
        if not block:
            lines = [
                re.sub(r"^\d+\.\s*", "", ln).strip("[]")
                for ln in response_text.splitlines()
                if re.match(r"^\d+\.", ln.strip())
            ]
            sub_queries = lines[:configurable.max_sub_queries]
        else:
            inner = block.group(1).strip()
            sub_queries = [
                re.sub(r"^\d+\.\s*", "", ln).strip("[]")
                for ln in inner.splitlines() if ln.strip()
            ]
            sub_queries = sub_queries[:configurable.max_sub_queries]
        
        return {
            "needs_decomposition": True,
            "sub_queries": sub_queries
        }
        
    except Exception as e:
        print(f"Error during query decomposition: {str(e)}")
        return {
            "needs_decomposition": False,
            "sub_queries": []
        }

# retrieve
def retrieve_raw_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    question = state["question"]
    sub_queries = state.get("sub_queries", [])
    all_queries = [question] + sub_queries
    is_original = [True] + [False] * len(sub_queries)

    # create/load the retriever
    retriever = create_retriever(
        top_k=cfg.top_k,
        subfolder=ROOT / "data/vectorstores",
        )

    # plain vector retrieval (no compression, no rerank)
    query_results = {}
    for q in all_queries:                       
        docs = retriever.invoke(q)
        query_results[q] = docs

    return {
        "docs_by_query": query_results,
        "is_original":   is_original,
    }

# rerank
def rerank_retrieved_docs(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)

    # 1. apply rerank fusion
    fused_docs = weighted_reciprocal_rank_fusion(
        state["docs_by_query"],
        state["is_original"],
        cfg.original_query_weight,
        cfg.sub_query_weight,
        cfg.smoothing_factor,
    )

    # 2. rerank with cohere (if API key is provided)
    if cfg.cohere_api_key:
        reranker = CohereRerank(
            cohere_api_key=cfg.cohere_api_key,
            model=cfg.rerank_model,
            top_n=cfg.rerank_top_n,
        )
        reranked_docs = reranker.compress_documents(
            fused_docs, 
            query=state["question"]
        )
    else:
        reranked_docs = fused_docs

    # 3. threshold filtering (remove all docs with a relevance score below the threshold)
    filtered_docs = [
        doc for doc in reranked_docs
        if doc.metadata.get("relevance_score", 0) >= cfg.score_threshold
    ][:cfg.rerank_top_n]

    return {"context": filtered_docs,
            "fused_docs": fused_docs,  
            "reranked": reranked_docs}
    

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
        input_schema = RagStateInput,
        output_schema = RagStateOutput_advanced,
        context_schema = Configuration,
    )

    # ── Decomposition → Raw Retrieval → RRF → Rerank → Threshold → Build Context → Rerank → Threshold → Generate
    builder.add_node("query_decomposition", query_decomposition_step)
    builder.add_node("retrieve", retrieve_raw_step)
    builder.add_node("rerank", rerank_retrieved_docs)
    builder.add_node("augment", augment_step)
    builder.add_node("generate", generate_step)

    builder.add_edge(START, "query_decomposition")
    builder.add_edge("query_decomposition", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "augment")
    builder.add_edge("augment", "generate")
    builder.add_edge("generate", END)

    return builder.compile(name="rag-agent-advanced")

# Create the graph instance
graph = create_rag_graph()