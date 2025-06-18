import streamlit as st
import uuid
from cog.agent import agent_runnable, get_session_history
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Personal Knowledge Manager", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📂 Personal Knowledge Manager")
    st.markdown("Your secure, offline assistant for local document management.")
    if st.button("🔄 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.header("💬 Chat with your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            # Use our updated agent implementation
            for chunk in agent_runnable["stream"](
                prompt, 
                session_id=st.session_state.session_id
            ):
                if "messages" in chunk:
                    last = chunk["messages"][-1]
                    # Handle different response formats safely
                    if isinstance(last, dict):
                        if "output" in last:
                            # Handle dictionary with output field
                            full_response += str(last["output"])
                            placeholder.markdown(full_response + "▌")
                        elif "tool" in last:
                            # Handle tool execution
                            tool_name = last.get('tool', 'tool')
                            placeholder.markdown(f"🔍 Searching your documents with `{tool_name}`...")
                            # Log additional info for debugging
                            print(f"Tool execution: {tool_name}, Arguments: {last.get('tool_input', 'unknown')}")
                            # If we've reached the end of this part of the chain, avoid early termination
                            if "end" in str(last).lower() or "finish" in str(last).lower():
                                print("Detected possible chain completion, continuing execution")
                    else:
                        # Try to get content from the message
                        try:
                            content = getattr(last, "content", None)
                            if content and isinstance(content, str):
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                        except Exception as e:
                            print(f"Error getting content: {e}")
                            # If anything fails, try to convert to string
                            try:
                                text = str(last)
                                if text and text != "None":
                                    full_response += text
                                    placeholder.markdown(full_response + "▌")
                            except Exception as e:
                                print(f"Error converting to string: {e}")
            
            # Display final response
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.error(f"❌ Error: {e}")
            full_response = f"Error: {e}"
        
        # Save the message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
