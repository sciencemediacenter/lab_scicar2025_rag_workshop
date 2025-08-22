from pathlib import Path
import sys
from typing import List
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic

from .configuration import Configuration

ROOT_DIR = Path(__file__).resolve().parents[3]      # repo root
PROMPT_DIR = ROOT_DIR / "data" / "prompts"
sys.path.append(str(PROMPT_DIR))

from rag_prompts import GEN_LLM_PROMPT

from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from .configuration import Configuration

def handle_anthropic_generation(
    question: str,
    context_docs: List[Document],
    cfg: Configuration,
    system_prompt: str = GEN_LLM_PROMPT,
) -> List[str]:
    """Generate answers using Anthropic model based on question and context.
    
    Args:
        question: The user's question
        context_docs: List of retrieved documents containing context
        cfg: Configuration object with model settings
        
    Returns:
        List of generated answers (multiple if cfg.num_answers > 1)
    """
    # Combine all documents into one context string
    context = "\n\n".join(doc.page_content for doc in context_docs)

    # augment system prompt with context
    system_prompt = GEN_LLM_PROMPT + f"\n{context}"
    
    # Initialize Anthropic client
    client = ChatAnthropic(
        model_name=cfg.default_model,
        temperature=cfg.temperature,
        anthropic_api_key=cfg.anthropic_api_key,
        max_tokens=cfg.max_tokens,
    )
    
    answers = []
    for _ in range(cfg.num_answers):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            response = client.invoke(messages, temperature=cfg.temperature)
            answers.append(response.content)
        except Exception as err:
            print(f"[Anthropic] Generation failed: {err}")
            answers.append("Entschuldigung, ich konnte keine Antwort generieren.")
            
    return answers
