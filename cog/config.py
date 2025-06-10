import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path    

class ModelProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

@dataclass
class ModelConfig:
    provider: ModelProvider
    name: str
    temperature: float = 0.7


QWEN25_14B = ModelConfig(
    provider=ModelProvider.OLLAMA,
    name="myaniu/qwen2.5-1m:14b",
    temperature=0.1
)


class Config:
    SEED = 42
    MODEL = QWEN25_14B
    OLLAMA_CONTEXT_WINDOW = 4096

    class Path:
        APP_HOME= Path(os.getenv("APP_HOME",  Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "data"
    
    class Server:
        HOST ="0.0.0.0"
        PORT=8000
        SSE_PATH = "/sse"
        TRANSPORT = "streamable-http"

        class Agent:
            MAX_ITERATIONS = 10
            
