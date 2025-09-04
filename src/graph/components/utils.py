import re
from typing import List, Dict
from langchain_core.documents import Document
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import calendar

from typing import List, Dict, Set, Tuple
from langchain.docstore.document import Document

############################
# Retrieval & Rank Fusion  #
############################

def weighted_reciprocal_rank_fusion(
    query_results: dict, 
    is_original_query: list, 
    original_query_weight: float = 2.0,
    sub_query_weight: float = 1.0,
    smoothing_factor: int = 60
) -> List[Document]:
    """
    Apply weighted Reciprocal Rank Fusion to combine results from multiple queries.
    Uses document's original UUID for tracking unique chunks.
    """
    chunk_scores = {}  # uuid -> score
    chunk_objects = {}  # uuid -> document
    
    # Process results for each query
    for query_idx, (query, docs) in enumerate(query_results.items()):
        query_weight = original_query_weight if is_original_query[query_idx] else sub_query_weight
        
        for rank, doc in enumerate(docs):
            # Get the document's unique identifier
            #doc_uuid = doc.metadata.get('id_') 
            doc_uuid = doc.id

            # Calculate score using weighted reciprocal rank formula
            score = query_weight * (1.0 / (smoothing_factor + rank + 1))
            
            if doc_uuid in chunk_scores:
                # If we've seen this chunk before, take the higher score
                chunk_scores[doc_uuid] = max(chunk_scores[doc_uuid], score)
            else:
                # First time seeing this chunk
                chunk_scores[doc_uuid] = score
                chunk_objects[doc_uuid] = doc
    
    # Sort chunks by final score
    sorted_chunks = sorted(
        [(doc_uuid, score) for doc_uuid, score in chunk_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Convert to list of documents
    combined_docs = [chunk_objects[doc_uuid] for doc_uuid, _ in sorted_chunks]
    
    # Add RRF scores to metadata
    for doc_uuid, score in chunk_scores.items():
        chunk_objects[doc_uuid].metadata['rrf_score'] = score
    
    return combined_docs

