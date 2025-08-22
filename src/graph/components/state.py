from __future__ import annotations
from typing import TypedDict, List, Dict
from typing_extensions import Annotated
from langchain_core.documents import Document
import operator

class RagStateInput(TypedDict, total=True):
    question: str

class RagState(TypedDict, total=False):
    # input 
    question: str

    # decomposition
    needs_decomposition: bool
    sub_queries: Annotated[List[str], operator.add]

    # raw retrieval output
    docs_by_query: Dict[str, List[Document]]
    is_original: List[bool]

    # fusion/rerank/filter 
    fused_docs: List[Document] # after applying rrf
    reranked: List[Document] # after reranking with cohere
    selected_docs: List[Document] # after threshold filtering

    # getting the full article from the chunks
    full_articles: List[Document]
    full_article_chunks: List[Document]

    # reranking and filtering the full articles
    reranked_articles: List[Document]
    
    # context for LLM
    context: List[Document]
    citation_chunks: List[Document]

    # augmenting the system prompt
    augmented_prompt: str

    # generation output
    answers: List[str]
    generation_metadata: Dict[str, str]

class RagStateOutput_naive(TypedDict, total=True):
    question: str
    context: List[Document]
    answers: List[str]

class RagStateOutput_advanced(TypedDict, total=True):
    question: str
    needs_decomposition: bool
    sub_queries: Annotated[List[str], operator.add]
    docs_by_query: Dict[str, List[Document]]
    fused_docs: List[Document] # after applying rrf
    reranked: List[Document] # after reranking with cohere
    context: List[Document]
    answers: List[str]
    