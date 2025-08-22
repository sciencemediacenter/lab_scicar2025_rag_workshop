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

def augment_context(
    question: str,
    context_docs: List[Document],
    system_prompt: str = GEN_LLM_PROMPT,
) -> str:
    """Combine retrieved context with system prompt.
    
    Args:
        question: The user's question
        context_docs: List of retrieved documents containing context
        system_prompt: Base system prompt to augment
        
    Returns:
        Augmented system prompt with context
    """
    # Combine all documents into one context string
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    # augment system prompt with context
    return f"\n{question}" + system_prompt + f"\n{context}"

def generate_answers(
    question: str,
    augmented_prompt: str,
    cfg: Configuration,
) -> List[str]:
    """Generate answers using Anthropic model based on augmented prompt.
    
    Args:
        question: The user's question
        augmented_prompt: System prompt augmented with context
        cfg: Configuration object with model settings
        
    Returns:
        List of generated answers (multiple if cfg.num_answers > 1)
    """
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
                {"role": "system", "content": augmented_prompt},
                {"role": "user", "content": question}
            ]
            response = client.invoke(messages, temperature=cfg.temperature)
            answers.append(response.content)
        except Exception as err:
            print(f"[Anthropic] Generation failed: {err}")
            answers.append("Entschuldigung, ich konnte keine Antwort generieren.")
            
    return answers
