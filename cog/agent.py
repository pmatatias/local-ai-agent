import operator
import asyncio
import os
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from cog.models import create_llm
from cog.config import QWEN25_14B, Config
from cog.models import Config
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# System prompt that describes the agent's capabilities and tools
SYSTEM_PROMPT = """You are a Personal Knowledge Manager assistant that helps users find, summarize, and analyze information from their local document collection.
You have access to the following tools to help users interact with their documents:
1. list_files(): Lists all files in the data directory with metadata including path, size, modification time, file extension, line count, and word count.
2. read_file(path): Reads the content of a specific file and returns its text. The path should be relative to the data directory.
3. extract_text(file_path): Extracts text from a file in the data directory and returns its content.
4. summarize_file(file_path): Generates a concise summary (3 sentences or less) of the specified file.
5. search(query, file_paths): Searches for the query in text files within the data directory. Returns a list of document chunks with their relevancy scores.
   - You can search all files by providing just the query
   - You can search specific files by providing a list of file paths
When users ask questions about their documents, use these tools to help them find relevant information. Always explain which tools you're using and why.
For example:
- If a user asks "What files do I have?", use list_files() to show all available files
- If a user asks "What's in my profile.md?", use read_file("profile.md") to display the content
- If a user asks "Summarize my profile document", use summarize_file("profile.md")
- If a user asks "Find information about Python", use search("Python") to search all documents
- If a user asks "Find Python in my code files", use search("Python", ["file1.py", "file2.py"])

Always provide helpful and accurate information based on the document content. If you can't find information in the documents, let the user know.
"""

# Create the language model
model = create_llm(Config.MODEL)
memory = MemorySaver()

# Tool implementations

def get_file_metadata(file_path):
    """Get metadata for a file."""
    path_obj = Path(file_path)
    stats = path_obj.stat()
    
    # Get line and word count for text files
    line_count = word_count = 0
    if path_obj.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                line_count = len(content.splitlines())
                word_count = len(content.split())
        except Exception:
            pass
    
    return {
        "path": str(file_path),
        "size": stats.st_size,
        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "extension": path_obj.suffix,
        "line_count": line_count,
        "word_count": word_count
    }

def list_files_tool(query: str = ""):
    """Lists all files in the data directory with metadata."""
    # Handle case where query is passed as a list
    if isinstance(query, list) and len(query) > 0:
        query = query[0]
    elif isinstance(query, list):
        query = ""
        
    print(f"Listing files with query: '{query}'")
    data_dir = Config.Path.DATA_DIR
    files = []
    
    for file_path in data_dir.glob('**/*'):
        if file_path.is_file():
            # If query is provided, filter files that match the query
            if not query or query.lower() in str(file_path).lower():
                files.append(get_file_metadata(file_path))
    
    # Add a message to indicate this is just a part of the overall execution
    print(f"Found {len(files)} files matching the query")
    result = json.dumps(files, indent=2)
    
    # Return the result with a clear indication this is just a list of files
    return result

def read_file_tool(path: str):
    """Reads the content of a specific file and returns its text."""
    # Handle case where path is passed as a list
    if isinstance(path, list) and len(path) > 0:
        path = path[0]
        
    file_path = Config.Path.DATA_DIR / path
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def extract_text_tool(file_path: str):
    """Extracts text from a file in the data directory and returns its content."""
    # Handle case where file_path is passed as a list
    if isinstance(file_path, list) and len(file_path) > 0:
        file_path = file_path[0]
    
    # For now, this is the same as read_file since we're only dealing with text files
    # In a real implementation, this would handle different file types (PDF, DOCX, etc.)
    return read_file_tool(file_path)

def summarize_file_tool(file_path: str):
    """Generates a concise summary of the specified file."""
    # Handle case where file_path is passed as a list
    if isinstance(file_path, list) and len(file_path) > 0:
        file_path = file_path[0]
    
    content = read_file_tool(file_path)
    
    # If the file wasn't found or had an error, return the error message
    if content.startswith("Error reading file:"):
        return content
    
    # Print debug info to help track the workflow
    print(f"Summarizing file: {file_path}")
    
    # Use the model to generate a summary
    messages = [
        SystemMessage(content="You are a summarization assistant. Create a concise summary (3 sentences or less) of the following text:"),
        HumanMessage(content=content)
    ]
    
    try:
        summary = model.invoke(messages).content
        # Add a marker to indicate this is the end of summarization for this file
        # but not the end of the entire agent execution
        result = f"Summary of {file_path}: {summary}"
        print(f"Successfully summarized {file_path}")
        return result
    except Exception as e:
        print(f"Error summarizing {file_path}: {e}")
        return f"Error summarizing {file_path}: {e}"

def search_tool(query: str, file_paths: Optional[List[str]] = None):
    """Searches for the query in text files within the data directory."""
    # Handle case where query is passed as a list
    if isinstance(query, list) and len(query) > 0:
        query = query[0]
    
    data_dir = Config.Path.DATA_DIR
    results = []
    
    # If no specific files are provided, search all text files
    if not file_paths:
        file_paths = [
            str(p.relative_to(data_dir))
            for p in data_dir.glob('**/*') 
            if p.is_file() and p.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']
        ]
    
    for file_path in file_paths:
        content = read_file_tool(file_path)
        # Simple search implementation - in a real system this would use embeddings and vector search
        if query.lower() in content.lower():
            # Find the context around the match
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    results.append({
                        "file": str(file_path),
                        "relevance": 0.8,  # Placeholder for actual relevance score
                        "context": context
                    })
                    break  # Just find the first match per file for simplicity
    
    return json.dumps(results, indent=2)

# Define the tools
tools = [
    Tool(
        name="list_files",
        func=list_files_tool,
        description="Lists all files in the data directory with metadata including path, size, modification time, file extension, line count, and word count."
    ),
    Tool(
        name="read_file",
        func=read_file_tool,
        description="Reads the content of a specific file and returns its text. The path should be relative to the data directory."
    ),
    Tool(
        name="extract_text",
        func=extract_text_tool,
        description="Extracts text from a file in the data directory and returns its content."
    ),
    Tool(
        name="summarize_file",
        func=summarize_file_tool,
        description="Generates a concise summary (3 sentences or less) of the specified file."
    ),
    Tool(
        name="search",
        func=search_tool,
        description="Searches for the query in text files within the data directory. Returns a list of document chunks with their relevancy scores."
    )
]

# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_openai_tools_agent(model, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=Config.Server.Agent.MAX_ITERATIONS,
    # memory is handled separately via RunnableWithMessageHistory
)

# Session history management
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create a session history for the given session ID."""
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# Define a dict-like object that wraps the agent_executor
class AgentRunnableWithStreaming:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.session_histories = {}
        
    def __getitem__(self, key):
        if key == "stream":
            return self.stream
        raise KeyError(f"Key {key} not found")
    
    def stream(self, input_text, session_id=None):
        """Stream the agent execution with history handling."""
        # Get or create the session history
        history = get_session_history(session_id) if session_id else ChatMessageHistory()
        
        # Add the new user message to history
        history.add_user_message(input_text)
        
        # Prepare inputs with history
        inputs = {
            "input": input_text,
            "chat_history": history.messages
        }
        
        # For tracking complete response
        final_output = ""
        last_chunk = None
        
        # Stream the execution
        for chunk in self.agent_executor.stream(inputs):
            last_chunk = chunk
            yield {"messages": [chunk]}
            
            # Track tool executions for debugging
            if isinstance(chunk, dict) and "tool" in chunk:
                print(f"Tool execution: {chunk.get('tool')}, Arguments: {chunk.get('tool_input')}")
                
                # If this is a summarize_file or list_files tool call, track it
                if chunk.get('tool') in ['summarize_file', 'list_files']:
                    # We'll continue executing even if it seems like the chain might be ending
                    print(f"Executing important tool: {chunk.get('tool')}")
            
            # Accumulate output if available
            if isinstance(chunk, dict) and "output" in chunk:
                final_output += str(chunk["output"])
            elif hasattr(chunk, "output") and getattr(chunk, "output"):
                final_output += str(getattr(chunk, "output"))
        
        # After execution completes, if there's an output, add it to history
        if last_chunk is not None:
            if isinstance(last_chunk, dict) and "output" in last_chunk:
                # Use the accumulated output instead of just the last chunk
                history.add_ai_message(final_output or str(last_chunk["output"]))
            elif hasattr(last_chunk, "output") and getattr(last_chunk, "output"):
                history.add_ai_message(final_output or str(getattr(last_chunk, "output")))
            elif final_output:
                # If we've accumulated output but it's not in the last chunk
                history.add_ai_message(final_output)
            else:
                # As a fallback, just convert the last chunk to a string
                history.add_ai_message(str(last_chunk))

# Create the agent_runnable with streaming capability
agent_runnable = AgentRunnableWithStreaming(agent_executor)
