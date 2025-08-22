from __future__ import annotations
import asyncio
from typing import Dict, List

from deepeval.metrics import (
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from tqdm.asyncio import tqdm_asyncio  # type: ignore

# internal helpers 
async def _score_single_answer(
    question: str,
    reference: str,
    contexts: List[str],
    answer: str,
    model: str = "gpt-4o-mini-2024-07-18",
) -> Dict[str, float]:
    """Return the three DeepEval scores for one answer."""
    tc = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=reference,
        retrieval_context=contexts,
    )

    # Create fresh metric objects – safer for asyncio than re-using stateful ones
    metrics = [
        ContextualRelevancyMetric(model=model, async_mode=True),
        AnswerRelevancyMetric(model=model, async_mode=True),
        FaithfulnessMetric(model=model, async_mode=True),
    ]

    # start all three measurements concurrently
    await asyncio.gather(*(m.a_measure(tc, _show_indicator=False) for m in metrics))

    return {
        "contextual_relevancy": metrics[0].score,
        "answer_relevancy":     metrics[1].score,
        "faithfulness":         metrics[2].score,
    }

async def _rate_item(item: dict, model: str, sem: asyncio.Semaphore) -> dict:
    """Score every answer field in one item dict; returns updated item."""
    question   = item.get("question") or item.get("query", "")
    reference  = item.get("gold_answer", "")
    contexts   = item.get("context", [])

    # Identify answer keys (either 'answer' or 'answer_no_X')
    answer_keys = (
        [k for k in item if k.startswith("answer_no_")] or
        (["answer"] if "answer" in item else [])
    )
    async def score_and_patch(ans_key: str):
        answer = item[ans_key]
        async with sem:          # limit concurrency (rate-limit safety) 
            scores = await _score_single_answer(
                question, reference, contexts, answer, model
            )
        # choose suffix
        suffix = "" if ans_key == "answer" else f"_no_{ans_key.split('_')[-1]}"
        for k, v in scores.items():
            item[f"{k}{suffix}"] = v

    await asyncio.gather(*(score_and_patch(k) for k in answer_keys))
    return item

# evaluation function
async def rate_dataset(
    data: List[dict],
    model: str = "gpt-4o-mini-2024-07-18",
    max_concurrency: int = 5,
) -> List[dict]:
    """
    Evaluate every answer for every record in `data`.
    Returns a *new* list with the same dicts mutated in place.
    """
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [_rate_item(rec, model, sem) for rec in data]
    return await tqdm_asyncio.gather(*tasks, desc="Scoring answers")

# sync wrapper (handy outside notebooks)
def rate_dataset_sync(
    data: List[dict],
    model: str = "gpt-4o-mini-2024-07-18",
    max_concurrency: int = 5,
) -> List[dict]:
    return asyncio.run(rate_dataset(data, model, max_concurrency))
