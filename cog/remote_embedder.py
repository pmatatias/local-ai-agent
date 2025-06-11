import requests
from langchain.embeddings.base import Embeddings

class RemoteOllamaEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def embed_documents(self, texts):
        response = requests.post(
            f"{self.endpoint}/api/embeddings",
            json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        return response.json()["data"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]