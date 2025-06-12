import os
import re
import textwrap
import time
from venv import create
from cog.config import QWEN25_14B, Config
from cog.models import create_llm
from langchain import text_splitter
import numpy as np

from ollama import embeddings
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from mcp.server.fastmcp import FastMCP
from cog.config import Config
from sqlalchemy import all_
from cog.config import embedding_config
from cog.remote_embedder import RemoteOllamaEmbeddings
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio


# Use the Config.Path.DATA_DIR from cog.config instead of defining a separate DATA_DIR
embedder = RemoteOllamaEmbeddings(
    endpoint=embedding_config.endpoint,
    model=embedding_config.model
)
class DocumentChunk(BaseModel):
    document_path: str
    text: str
    relevancy_score: float = Field(..., description="Relevancy score of the chunk based on search query")

class File(BaseModel):
    path: str
    size: int
    modified_time: float
    created_time: float
    extension: str
    line_count: int | None
    word_count: int | None

class SearchResult(BaseModel):
    content: str
    source: str
    score: float

SUMMARIZE_PROMPT = textwrap.dedent("""
/no_think
Please provide a concise summary in 3 sentences or lessof the following text:

{text}

Summary:
""").strip()

summary_model = create_llm(QWEN25_14B)

mcp = FastMCP()

app = FastAPI()

@app.get("/events")
async def sse_endpoint(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            yield f"data: Hello at {time.time()}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def _build_vector_store():
    loader = DirectoryLoader(
        str(Config.Path.DATA_DIR), glob="**/*.{txt,md,py,json}", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = FastEmbedEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

@mcp.tool()
def list_files() -> list[File]:
    """
    List all files in the data directory with metadata.
    File information includes:
    - path\n
    - size (in bytes)\n
    - modified_time (timestamp)\n
    - created_time (timestamp)\n
    - extension (file type)\n
    - line_count (number of lines, if applicable)\n
    - word_count (number of words, if applicable)\n
    This tool scans the data directory and returns a list of File objects.   
    """
    files = []
    for root, _, fnames in os.walk(Config.Path.DATA_DIR):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, Config.Path.DATA_DIR)
            try:
                stat = os.stat(fpath)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                files.append(File(
                    path=relpath, size=stat.st_size, modified_time=stat.st_mtime,
                    created_time=stat.st_ctime, extension=os.path.splitext(fname)[1],
                    line_count=len(content.splitlines()), word_count=len(content.split())
                ))
            except Exception: pass
    return files

@mcp.tool()
def extract_text (file_path:str)-> str:
    """
    Extract text from a file in the data directory.
    This tool reads the content of a specified file and returns its text.
    """
    return (Config.Path.DATA_DIR / file_path).read_text()

@mcp.tool()
def summarize_file(file_path:str)-> str:
    """
    Generate a concise summary of the specified file.
    This tool reads the content of a file and generates a summary in 3 sentences or less.
    """
    if not (Config.Path.DATA_DIR / file_path).exists():
        return f"File not found: {file_path}"
    
    
    file_content = (Config.Path.DATA_DIR/ file_path).read_text()
    response = summary_model.invoke(SUMMARIZE_PROMPT.format(text=file_content))
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    else:
        content = str(content)
    return re.sub(r"<[^>]*>", "", content).strip()

@mcp.tool()
def read_file(path: str = Field(...)) -> str:
    full_path = os.path.abspath(os.path.join(Config.Path.DATA_DIR, path))
    if not full_path.startswith(str(Config.Path.DATA_DIR)):
        return "Security alert: Invalid path."
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def search(query: str , file_paths: list[str]=[]) -> list[DocumentChunk]:
    """
    Search for a query in the text files in the data directory.
    Returns a list of DocumentChunk objects containing the text and relevancy score.
    """
    text_splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    # get file to search
    if file_paths:
        # Search only specified files
        files_to_search =[
            Config.Path.DATA_DIR/ path
            for path in file_paths if (Config.Path.DATA_DIR / path).exists()
        ]
    else:
        # Search all text files in the data directory
        files_to_search = [
            f
            for f in Config.Path.DATA_DIR.glob("**/*.{txt,md,py,json}")
        ]

    #collect all chunks with their source files
    all_chunks = []
    chunk_texts = []
    
    # Process each file
    for file_path in files_to_search:
        try:
            # Read file content
            file_content = file_path.read_text(encoding="utf-8", errors="ignore")
            
            # Create chunks using semantic chunker
            chunks = text_splitter.create_documents([file_content])
            
            # Process each chunk
            for chunk in chunks:
                relative_path = str(file_path.relative_to(Config.Path.DATA_DIR))
                all_chunks.append({
                    "document_path": relative_path,
                    "text": chunk.page_content,
                })
                chunk_texts.append(chunk.page_content)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    # If no chunks were found, return empty list
    if not all_chunks:
        return []
    
    # Calculate relevancy scores using embeddings
    query_embedding = embedder.embed_query(query)
    chunk_embeddings = embedder.embed_documents(chunk_texts)
    # similarities = np.dot(chunk_embeddings, query_embedding) 
    # similarity_ind= np.argsort(similarities)[::-1]  


    
    # Calculate cosine similarity for relevancy scores
    relevancy_scores = []
    for chunk_embedding in chunk_embeddings:
        # if similarity_ind.size <0.1:
        #     continue
        # Calculate cosine similarity between query and chunk
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        relevancy_scores.append(float(similarity))
    
    # Create DocumentChunk objects with relevancy scores
    results = []
    for idx, chunk_info in enumerate(all_chunks):
        results.append(DocumentChunk(
            document_path=chunk_info["document_path"],
            text=chunk_info["text"],
            relevancy_score=relevancy_scores[idx]
        ))
    
    # Sort results by relevancy score (highest first)
    results.sort(key=lambda x: x.relevancy_score, reverse=True)
    
    return results

if __name__ == "__main__":
    print("Running MCP server...")
    mcp.run()
