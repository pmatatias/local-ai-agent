from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from cog.config import Config, ModelConfig, ModelProvider

def create_llm(model_config: ModelConfig) -> BaseChatModel:
    """Create a language model instance based on the provided configuration."""
    if model_config.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            base_url= model_config.base_url,         
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_config.provider}")