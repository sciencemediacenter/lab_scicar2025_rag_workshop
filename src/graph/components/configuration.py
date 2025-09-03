import os
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# Load environment variables and print status immediately
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Print API key status at module level
def get_api_keys_status():
    """Get the status of all required API keys."""
    api_keys_status = {
        'OpenAI': bool(OPENAI_API_KEY),
        'Anthropic': bool(ANTHROPIC_API_KEY),
        'Cohere': bool(COHERE_API_KEY)
    }
    
    return {
        "all_keys_present": all(api_keys_status.values()),
        "status": api_keys_status,
        "missing": [name for name, present in api_keys_status.items() if not present]
    }

class Configuration(BaseModel):
    """The configuration for the RAG agent."""
    
    # Provider and model settings
    provider: str = Field(
        default="anthropic",
        metadata={"description": "LLM provider (openai or anthropic)"}
    )
    answer_gen_model: str = Field(
        default=None,
        metadata={"description": "Model name (provider-specific, uses default if None)"}
    )
    decomposition_dec_model: str = Field(
        default="gpt-4.1-nano-2025-04-14",
        metadata={"description": "Model for query decomposition decision"}
    )
    decomposition_gen_model: str = Field(
        default="gpt-4o-mini-2024-07-18",
        metadata={"description": "Model for query decomposition generation"}
    )
    rerank_model: str = Field(
        default="rerank-v3.5",
        metadata={"description": "Cohere rerank model"}
    )
    
    # Retrieval settings
    top_k: int = Field(
        default=10,
        metadata={"description": "Number of chunks to retrieve"}
    )

    # Generation settings
    num_answers: int = Field(
        default=1,
        metadata={"description": "Number of answers to generate"}
    )
    temperature: float = Field(
        default=0.5,
        metadata={"description": "Temperature parameter"}
    )
    max_tokens: int = Field(
        default=2048,
        metadata={"description": "Maximum number of output tokens in the response"}
    )

    # API keys
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv('OPENAI_API_KEY'),
        metadata={"description": "OpenAI API key"}
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv('ANTHROPIC_API_KEY'),
        metadata={"description": "Anthropic API key"}
    )
    cohere_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv('COHERE_API_KEY'),
        metadata={"description": "Cohere API key for reranking"}
    )

    # Citation settings
    use_citations: bool = Field(
        default=True,
        metadata={"description": "Whether to use document citations (works only with Anthropic)"}
    )
    
    # Reranking settings
    rerank_top_n: int = Field(
        default=10,
        metadata={"description": "Number of docs after reranking"}
    )
    score_threshold: float = Field(
        default=0.15,
        metadata={"description": "Minimum relevance score threshold"}
    )
    
    # Query decomposition settings
    use_query_decomposition: bool = Field(
        default=True,
        metadata={"description": "Whether to use query decomposition"}
    )
    max_sub_queries: int = Field(
        default=3,
        metadata={"description": "Maximum number of sub-queries"}
    )
    original_query_weight: float = Field(
        default=2.0,
        metadata={"description": "Weight for original query in RRF"}
    )
    sub_query_weight: float = Field(
        default=1.0,
        metadata={"description": "Weight for sub-queries in RRF"}
    )
    smoothing_factor: int = Field(
        default=60,
        metadata={"description": "Smoothing factor for RRF"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)
    
    @property
    def default_model(self) -> str:
        """Get the default model for the current provider."""
        defaults = {
            "openai": "gpt-4o-mini-2024-07-18",
            "anthropic": "claude-3-5-haiku-20241022"
        }
        return self.answer_gen_model or defaults.get(self.provider, defaults["anthropic"])
    
    @model_validator(mode="after")
    def _check_api_keys(self):
        """Validate that required API keys are present."""
        # If no API key was provided in constructor, try to get it from environment
        if self.provider == "openai":
            self.openai_api_key = self.openai_api_key or os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when provider='openai'")
        elif self.provider == "anthropic":
            self.anthropic_api_key = self.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when provider='anthropic'")
        return self

    def __init__(self, **data):
        # If API keys aren't provided in data, get them from environment
        if 'anthropic_api_key' not in data and data.get('provider') == 'anthropic':
            data['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if 'openai_api_key' not in data and data.get('provider') == 'openai':
            data['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if 'cohere_api_key' not in data:
            data['cohere_api_key'] = os.getenv('COHERE_API_KEY')
        
        super().__init__(**data)