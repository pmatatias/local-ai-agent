import operator
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain.agents import ToolExecutor
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt 
from langchain_mcp_adapters.client  import MultiServerMCPClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

mcp_client = MultiServerMCPClient()
tools = mcp_client.get_tools()
tool_executor = ToolExecutor(tools)
model = ChatOllama(model="llama3:8b")  # You must have it pulled locally

model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState):
    last = state["messages"][-1]
    tool_call = ToolInvocation(tool=last.tool_calls[0]["name"], tool_input=last.tool_calls[0]["args"])
    result = tool_executor.invoke(tool_call)
    return {"messages": [ToolMessage(content=str(result), tool_call_id=last.tool_calls[0]["id"])]}

def should_continue(state: AgentState):
    return "continue" if state["messages"][-1].tool_calls else "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

compiled = workflow.compile()
chat_memory = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_memory:
        chat_memory[session_id] = ChatMessageHistory()
    return chat_memory[session_id]

agent_runnable = RunnableWithMessageHistory(
    compiled, get_session_history, input_messages_key="messages", history_messages_key="messages"
)
