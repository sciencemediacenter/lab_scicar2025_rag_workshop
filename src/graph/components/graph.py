# graph_agent/graph.py
import os
from dotenv import load_dotenv
from langsmith import Client
import langchain

# Load environment variables first
load_dotenv()

# Set up weaviate client
WEAVIATE_HOST = os.getenv('HOST')
WEAVIATE_PORT = os.getenv('PORT')
WEAVIATE_GRPC_PORT = os.getenv('GRPC_PORT')

import re
from typing import List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

from langsmith import traceable
import pandas as pd
import pathlib

from .state import RagStateInput, RagState, RagStateOutput
from .configuration import Configuration
from .utils import (
    weighted_reciprocal_rank_fusion,
    build_full_articles,
    filter_and_rescore,
    inject_pub_stamp,
)
from .llm_handlers import (
    handle_anthropic_generation,
    handle_openai_generation,
    has_no_info_indicators,
)
from .retriever import (
    create_weaviate_retriever
)

from .prompts import (
    NO_ANSWER_TEMPLATE,
    NO_INFO_INDICATORS,
    DECOMP_PROMPT,
    CLASSIFY_DECOMP_PROMPT
)

# create/load the weaviate retriever and the weaviate client
base_retriever, weaviate_client = create_weaviate_retriever(
    top_k=10,                      
    embedding_model="text-embedding-3-large",
)
# check client status
def get_weaviate_status():
    return {
        "is_ready": weaviate_client is not None,
        "client": weaviate_client
    }

@traceable(name="query_decomposition")
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

@traceable(name="retrieval_step_all")
def retrieve_raw_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    question = state["question"]
    sub_queries = state.get("sub_queries", [])
    all_queries = [question] + sub_queries
    is_original = [True] + [False] * len(sub_queries)

    # plain vector retrieval (no compression, no rerank)
    query_results = {}
    for q in all_queries:                       
        docs = base_retriever.invoke(q)
        query_results[q] = docs

    return {
        "docs_by_query": query_results,
        "is_original":   is_original,
    }

@traceable(name="apply_reciprocal_rank_fusion")
def rrf_fusion_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    fused = weighted_reciprocal_rank_fusion(
        state["docs_by_query"],
        state["is_original"],
        cfg.original_query_weight,
        cfg.sub_query_weight,
        cfg.smoothing_factor,
    )
    return {"fused_docs": fused}

@traceable(name="rerank_with_cohere")
def rerank_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)

    if not cfg.cohere_api_key:                  
        return {"reranked": state["fused_docs"]}

    reranker = CohereRerank(
        cohere_api_key = cfg.cohere_api_key,
        model = cfg.rerank_model,
        top_n = cfg.rerank_top_n,
    )
    reranked = reranker.compress_documents(
        state["fused_docs"], query=state["question"]
    )
    return {"reranked": reranked}

@traceable(name="apply_threshold_filter")
def threshold_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)
    kept = [d for d in state["reranked"]
            if d.metadata.get("relevance_score",0) >= cfg.score_threshold]
    return {"selected_docs": kept[:cfg.rerank_top_n]}

@traceable(name="build_full_context")
def context_step(state: RagState, config: RunnableConfig):
    
    full_articles, citation_chunks = build_full_articles(
        state["selected_docs"], 
        weaviate_client=weaviate_client,
    )
        
    return {"full_articles": full_articles,
            "full_article_chunks": citation_chunks}

@traceable(name="rerank_full_articles")
def rerank_full_article_step(state: RagState, config: RunnableConfig):
    cfg = Configuration.from_runnable_config(config)

    reranker = CohereRerank(
        cohere_api_key = cfg.cohere_api_key,
        model = cfg.rerank_model,
        top_n = len(state["full_articles"]),
    )
    reranked_articles = reranker.compress_documents(
        state["full_articles"], 
        query=state["question"]
    )
    return {"reranked_articles": reranked_articles}

@traceable(name="filter_articles_by_score")
def filter_articles_step(state: RagState, config: RunnableConfig):
    """Filter articles based on normalized relevance scores and weighted recency and update citation chunks accordingly."""
    cfg = Configuration.from_runnable_config(config)
    
    reranked_articles = state["reranked_articles"]
    citation_chunks = state["full_article_chunks"]
    
    # Early return if no articles to process
    if not reranked_articles:
        print("Debug: No articles to filter")
        return {
            "context": [],
            "citation_chunks": []
        }

    context, citation_chunks = filter_and_rescore(
        docs = reranked_articles, 
        chunks = citation_chunks,
        relevance_floor = cfg.score_threshold,
        alpha           = cfg.time_alpha,
        max_age_days    = cfg.max_age_days,
    )

    inject_pub_stamp(context) 
    
    return {
        "context": context,
        "citation_chunks": citation_chunks
    }

@traceable(name="generate_final_answer")
def generate_step(state: RagState, config: RunnableConfig):
    """Generate an answer based on the retrieved context."""
    configurable = Configuration.from_runnable_config(config)
    question = state["question"]
    context = state["context"]
    citation_chunks = state["citation_chunks"]
    
    try: # Anthropic
        if configurable.provider == "anthropic":
            answers = handle_anthropic_generation(
                question=question, 
                context_docs=context, 
                citation_chunks=citation_chunks, 
                cfg=configurable
            )
        else:  # OpenAI
            answers = handle_openai_generation(
                question=question, 
                context_docs=context, 
                cfg=configurable
            )
        
        # Check for no-information indicators
        if any(has_no_info_indicators(ans, NO_INFO_INDICATORS) for ans in answers):
            answers = [NO_ANSWER_TEMPLATE] * configurable.num_answers
        
        return {
            "answers": answers
        }
        
    except Exception as e:
        print(f"Error in generate step: {str(e)}")
        return {
            "answers": [NO_ANSWER_TEMPLATE] * configurable.num_answers
        }

def create_rag_graph():
    builder = StateGraph(
        RagState,
        input = RagStateInput,
        output = RagStateOutput,
        config_schema = Configuration,
    )

    # ── Decomposition → Raw Retrieval → RRF → Rerank → Threshold → Build Context → Rerank → Threshold → Generate
    builder.add_node("query_decomposition", query_decomposition_step)
    builder.add_node("retrieve", retrieve_raw_step)
    builder.add_node("apply_rrf", rrf_fusion_step)
    builder.add_node("apply_rerank", rerank_step)
    builder.add_node("apply_filter", threshold_step)
    builder.add_node("build_context", context_step)
    builder.add_node("apply_rerank_full_articles", rerank_full_article_step)
    builder.add_node("filter_reranked_full_articles", filter_articles_step)
    builder.add_node("get_answer", generate_step)

    builder.add_edge(START, "query_decomposition")
    builder.add_edge("query_decomposition", "retrieve")
    builder.add_edge("retrieve", "apply_rrf")
    builder.add_edge("apply_rrf", "apply_rerank")
    builder.add_edge("apply_rerank", "apply_filter")
    builder.add_edge("apply_filter", "build_context")
    builder.add_edge("build_context", "apply_rerank_full_articles")
    builder.add_edge("apply_rerank_full_articles", "filter_reranked_full_articles")
    builder.add_edge("filter_reranked_full_articles", "get_answer")
    builder.add_edge("get_answer", END)

    return builder.compile(name="rag-agent-v4")

# Create the graph instance
graph = create_rag_graph()