import streamlit as st
import uuid
from cog.agent import agent_runnable
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Private AI Assistant", page_icon="🤖", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📂 Local AI Assistant")
    st.markdown("Secure, offline assistant for your documents.")
    if st.button("🔄 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

st.header("💬 Chat with your files")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in agent_runnable.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config={"configurable": {"session_id": st.session_state.session_id}}
            ):
                if "messages" in chunk:
                    last = chunk["messages"][-1]
                    if last.tool_calls:
                        placeholder.markdown(f"🛠 Calling `{last.tool_calls[0]['name']}`...")
                    elif isinstance(last.content, str):
                        full_response += last.content
                        placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.error(f"❌ Error: {e}")
            full_response = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": full_response})
