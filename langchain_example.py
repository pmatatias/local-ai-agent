"""Example script demonstrating how to use langchain_core.language_models."""
from langchain_core.language_models import BaseChatModel, BaseLLM, BaseLanguageModel
from langchain_core.messages import HumanMessage
from typing import List, Optional, Any, Dict
from langchain_core.outputs import ChatGeneration, ChatResult


class SimpleDemoChatModel(BaseChatModel):
    """A simple demonstration implementation of BaseChatModel."""
    
    def _generate(
        self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a chat result from the given messages."""
        response = "This is a demo response from SimpleDemoChatModel"
        generation = ChatGeneration(message=HumanMessage(content=response))
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self, messages: List[Any], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a chat result from the given messages asynchronously."""
        return self._generate(messages, stop, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return "simple_demo_chat_model"


def main():
    """Main function demonstrating langchain_core usage."""
    # Print available classes
    print("Available base classes in langchain_core.language_models:")
    print("- BaseChatModel: Base class for chat models")
    print("- BaseLLM: Base class for language models")
    print("- BaseLanguageModel: Base class for all language models\n")
    
    # Create an instance of our demo chat model
    model = SimpleDemoChatModel()
    
    # Demonstrate usage
    messages = [HumanMessage(content="Hello, how are you?")]
    result = model.invoke(messages)
    print(f"Model response: {result.content}")


if __name__ == "__main__":
    main()
