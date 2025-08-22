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
from weaviate.classes.query import Filter, QueryReference

############################
# Full Document Processing #
############################

#  CONFIG 
COLL_NAME = "Text"
FIELDS_TO_GET = [
    "text", "story_id", "id_", "element_order",
    "text_type_story_title", "text_type_story_description",
    "text_type_bullet_point", "text_type_paragraph",
    "text_type_statement", "text_type_statement_complete",
    "text_type_statement_restructured", "text_type_question",
    "text_type_intertitle", "text_type_statements_description",    
]

#  UTILITY 
def fetch_sorted(
    collection,
    filter_: Filter,
    limit: int = 10_000,
    return_refs: List[QueryReference] | None = None,
):
    """Wrapper: fetch objects and sort by element_order (None -> 0)."""
    res = collection.query.fetch_objects(
        filters=filter_,
        limit=limit,
        return_properties=FIELDS_TO_GET,
        return_references=return_refs or [],
    )
    return sorted(res.objects, key=lambda o: o.properties.get("element_order") or 0)

#  MAIN BUILDER
def build_full_articles(
    selected_docs: List[Document],
    weaviate_client,
) -> Tuple[List[Document], List[Document]]:
    """
    Build one full-article document per story **and** collect all teaser /
    statement chunks (deduplicated) for citation use.

    Returns
    -------
    full_articles : List[Document]
    citation_chunks : List[Document]
    """    
    texts = weaviate_client.collections.get(COLL_NAME)

    # 1) which stories did the retriever surface?
    story_ids: Set[str] = {
        doc.metadata["story_id"]
        for doc in selected_docs
        if "story_id" in doc.metadata
    }

    # 2) which statement chunks (id_) appeared per story?
    story_id_to_stmt_chunk_ids: Dict[str, Set[str]] = {}
    for doc in selected_docs:
        if doc.metadata.get("text_type_statement"):
            story_id_to_stmt_chunk_ids.setdefault(
                doc.metadata["story_id"], set()
            ).add(doc.metadata["id_"])

    full_articles: List[Document] = []
    citation_chunks: List[Document] = []
    seen_chunk_ids: Set[str] = set()          # for deduplication

    for story_id in story_ids:

        parts: List[str] = ["=== ARTIKEL START ==="]

        # TITLE
        filt_title = (
            Filter.by_property("story_id").equal(story_id)
            & Filter.by_property("text_type_story_title").equal(True)
        )
        title_obj = fetch_sorted(texts, filt_title, limit=1)
        title = title_obj[0].properties["text"] if title_obj else "(Kein Titel)"
        parts.append(f"TITEL: {title}")

        # SHORT DESCRIPTION
        filt_desc = (
            Filter.by_property("story_id").equal(story_id)
            & Filter.by_property("text_type_story_description").equal(True)
        )
        desc_obj = fetch_sorted(texts, filt_desc, limit=1)
        short_desc = (
            desc_obj[0].properties["text"] if desc_obj else "(Keine Kurzbeschreibung)"
        )
        parts.append(f"KURZBESCHREIBUNG: {short_desc}")

        # BULLET POINTS 
        filt_bullets = (
            Filter.by_property("story_id").equal(story_id)
            & Filter.by_property("text_type_bullet_point").equal(True)
        )
        bullets = fetch_sorted(texts, filt_bullets)
        if bullets:
            parts.append("=== STICHPUNKTE ===")
            parts.extend([b.properties["text"] for b in bullets])

        # TEASER 
        filt_teaser = (
            Filter.by_property("story_id").equal(story_id)
            & Filter.by_property("text_type_paragraph").equal(True)
            & Filter.by_property("text_type_statement").not_equal(True)
            & Filter.by_property("text_type_statement_complete").not_equal(True)
            & Filter.by_property("text_type_statement_restructured").not_equal(True)
            & Filter.by_property("text_type_question").not_equal(True)
        )
        teaser_paragraphs = fetch_sorted(texts, filt_teaser)
        if teaser_paragraphs:
            parts.append("=== HAUPTTEXT ===")
            for p in teaser_paragraphs:
                # add to article
                parts.append(p.properties["text"])
                # add to citation set
                cid = p.properties["id_"]
                if cid not in seen_chunk_ids:
                    citation_chunks.append(
                        Document(
                            page_content=p.properties["text"],
                            metadata={
                                "story_id": story_id,
                                "title": title,          
                                "id_": cid,
                                "text_type_statement": False,
                            },
                        )
                    )
                    seen_chunk_ids.add(cid)

        # STATEMENTS 
        stmt_chunk_ids = story_id_to_stmt_chunk_ids.get(story_id, set())
        expert_ids: Set[str] = set()

        # 3a) get contact_id/expert_id for each retrieved statement chunk
        if stmt_chunk_ids:
            single_person_ref = QueryReference.MultiTarget(
                link_on="hasPerson",
                target_collection="Person",
                return_properties=["contact_id"],
            )
            for chunk_id in stmt_chunk_ids:
                filt_chunk = (
                    Filter.by_property("story_id").equal(story_id)
                    & Filter.by_property("id_").equal(chunk_id)
                    & Filter.by_property("text_type_statement").equal(True)
                )
                obj = texts.query.fetch_objects(
                    filters=filt_chunk,
                    limit=1,
                    return_properties=[],
                    return_references=[single_person_ref],
                ).objects[0]
                ref_objs = obj.references["hasPerson"].objects
                if ref_objs:
                    expert_ids.add(ref_objs[0].properties["contact_id"])

        # 3b) get all statement paragraphs and create full statement per expert
        expert_to_statements: Dict[str, List[str]] = {}
        if expert_ids:
            ref_contact = QueryReference.MultiTarget(
                link_on="hasPerson",
                target_collection="Person",
                return_properties=["contact_id", "first_name", "last_name"],
            )
            stmt_filter = (
                Filter.by_property("story_id").equal(story_id)
                & Filter.by_property("text_type_statement").equal(True)
                & Filter.by_property("text_type_statement_restructured").not_equal(True)
                & Filter.by_property("text_type_statement_complete").not_equal(True)
            )
            stmt_chunks = fetch_sorted(texts, stmt_filter, return_refs=[ref_contact])

            tmp: Dict[str, List[Tuple[int, str]]] = {}
            for chunk in stmt_chunks:
                ref_objs = chunk.references["hasPerson"].objects
                if not ref_objs:
                    continue
                cid = ref_objs[0].properties["contact_id"]
                if cid not in expert_ids:
                    continue

                order = chunk.properties.get("element_order", 0) or 0
                tmp.setdefault(cid, []).append((order, chunk.properties["text"]))

            for cid, pieces in tmp.items():
                pieces.sort(key=lambda t: t[0])                 # order 0,1,2,…
                statement_text = "\n".join(text for _, text in pieces)
                expert_to_statements.setdefault(cid, []).append(statement_text)

        # append to article
        if expert_to_statements:
            parts.append("=== STATEMENTS ===")
            for cid, paras in expert_to_statements.items():
                parts.append(f"\n--- EXPERT {cid} ---")
                parts.extend(paras)

        # 3c) simple text chunks for citation (expert-filtered)
        if expert_ids:
            ref_contact_stmt = QueryReference.MultiTarget(
                link_on="hasPerson",
                target_collection="Person",
                return_properties=["contact_id", "display_name"],
            )

            stmt_filter = (
                Filter.by_property("story_id").equal(story_id)
                & Filter.by_property("text_type_statement").equal(True)
                & Filter.by_property("text_type_statement_restructured").not_equal(True)
                & Filter.by_property("text_type_statement_complete").not_equal(True)
            )

            stmt_chunks = fetch_sorted(texts, stmt_filter, return_refs=[ref_contact_stmt])

            for chunk in stmt_chunks:
                ref_objs = chunk.references["hasPerson"].objects
                if not ref_objs:
                    continue
                cid = ref_objs[0].properties["contact_id"]
                if cid not in expert_ids:
                    continue                     # only the surfaced experts

                expert_name = ref_objs[0].properties["display_name"]

                cid_ = chunk.properties["id_"]
                if cid_ in seen_chunk_ids:
                    continue                     # dedup across teaser & statements

                citation_chunks.append(
                    Document(
                        page_content=chunk.properties["text"],
                        metadata={
                            "story_id": story_id,
                            "title": title,
                            "id_": cid_,
                            "expert_id": cid,
                            "expert_name": expert_name,
                            "text_type_statement": True,
                        },
                    )
                )
                seen_chunk_ids.add(cid_)

        # FINISH FULL ARTICLE
        parts.append("=== ARTIKEL ENDE ===")
        page_content = "\n\n".join(parts)

        # max relevance among this story’s retrieved chunks
        max_relevance = max(
            (
                doc.metadata.get("relevance_score", 0)
                for doc in selected_docs
                if doc.metadata.get("story_id") == story_id
            ),
            default=0,
        )

        full_articles.append(
            Document(
                page_content=page_content,
                metadata={
                    "story_id": story_id,
                    "title": title,
                    "short_description": short_desc,
                    "original_relevance": max_relevance,
                    "expert_ids": sorted(expert_ids),
                    "number_of_statements": len(expert_ids),
                    "word_length": len(page_content)
                },
            )
        )

    return full_articles, citation_chunks

############################
# Time & Date Information  #
############################

ORDINAL_DE = {1: "Neuester Beitrag",
              2: "Zweitneuester Beitrag",
              3: "Drittneuester Beitrag"}

def inject_pub_stamp(docs):
    """
    Mutates each Document.page_content in place:
      - inserts a banner just after '=== ARTIKEL START ==='
      - format:  [1. Neuester Beitrag – Veröffentlicht: Mai 2024]
    Assumes docs are already sorted by combined_score (best first).
    """
    for idx, d in enumerate(docs, start=1):
        pub_dt = extract_pub_date(d.metadata["story_id"])
        month  = calendar.month_name[pub_dt.month]
        year   = pub_dt.year
        label  = ORDINAL_DE.get(idx, f"{idx}. Beitrag")
        banner = f"[{label} – Veröffentlicht: {month} {year}]"

        if "=== ARTIKEL START ===" in d.page_content:
            head, body = d.page_content.split("=== ARTIKEL START ===", 1)
            d.page_content = (
                "=== ARTIKEL START ===\n\n"
                + banner + "\n\n"
                + body.lstrip()
            )
    return docs

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

#################################
# Post-retrieval rerank/filter  #
#################################

def extract_pub_date(story_id: str) -> datetime:
    """
    story_id = YYDDD   where YY = 00‑99 (→ 2000‑2099) and
                                DDD = ordinal of publication that year (001‑999)
    """
    sid = int(story_id)
    year   = 2000 + sid // 1000                # 25 → 2025
    ordinal = sid % 1000 or 1                  
    day_of_year = int(round((ordinal - 1) * 1.5))
    return datetime(year, 1, 1) + timedelta(days=day_of_year)

def blended_score(rel_norm: np.ndarray,
                  story_ids: List[str],
                  *,
                  alpha: float,
                  max_age_days: int) -> np.ndarray:
    """α·relevance  +  (1‑α)·recency  → vector of floats (0‑1)."""
    now  = datetime.now()
    ages = np.array([(now - extract_pub_date(sid)).days for sid in story_ids])
    ages        = np.clip(ages, 0, max_age_days)
    time_norm   = 1 - ages / max_age_days
    return alpha * rel_norm + (1 - alpha) * time_norm

def filter_and_rescore(
        docs: List,                         # reranked articles
        chunks: List,                       # full_article_chunks
        *,
        relevance_floor: float = 0.15,
        alpha: float = 0.80,
        max_age_days: int = 365 * 5,
) -> Tuple[List, List]:
    """
    1. Normalise Cohere scores.
    2. Remove irrelevant articles.
    3. Add recency weight & sort.
    4. Return (kept_docs, kept_chunks).
    """
    if not docs:
        return [], []

    # normalize cohere relevance score
    rel = np.array([d.metadata["relevance_score"] for d in docs], dtype=float)
    rel_norm = rel / rel.max()

    # keep only articles > score_threshold
    keep_mask   = rel_norm >= relevance_floor
    kept_docs   = [d for d, m in zip(docs, keep_mask) if m]
    rel_norm_kept = rel_norm[keep_mask]

    if not kept_docs:
        return [], []

    # add recency weight to the scores
    story_ids = [d.metadata["story_id"] for d in kept_docs]
    combined  = blended_score(rel_norm_kept, story_ids,
                              alpha=alpha, max_age_days=max_age_days)

    for d, s in zip(kept_docs, combined):
        d.metadata["combined_score"] = float(s)

    kept_docs.sort(key=lambda d: d.metadata["combined_score"], reverse=True)

    # remove irrelevant chunks
    kept_ids = {d.metadata["story_id"] for d in kept_docs}
    kept_chunks = [c for c in chunks if c.metadata["story_id"] in kept_ids]

    return kept_docs, kept_chunks

