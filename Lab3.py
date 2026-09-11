import pymupdf
import streamlit as st
from openai import OpenAI, AuthenticationError

# Hide the sidebar and the collapse/expand toggle button
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Positioning header
st.markdown(
    """
    <style>
        .st-key-app_header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            background-color: var(--background-color);
            padding: 3.5rem 1rem 1rem 1rem;
            text-align: center;
        }
        .st-key-app_header h1,
        .st-key-app_header p {
            text-align: center;
        }
        .stMainBlockContainer {
            padding-top: 9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Show title and description.
with st.container(key="app_header"):
    st.title("📄 Mel's Chatbot")
    st.caption("• Let's chat.")

# Retrieve API key and create an OpenAI client.
if "client" not in st.session_state:
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.client = OpenAI(api_key=openai_api_key)

# Checking if API key is valid.
try:
    st.session_state.client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

# System prompt
system_prompt = {
    "role": "system",
    "content": (
        "Explain everything simply enough that a 10 year old could understand it. Use short sentences, everyday words, "
        "concrete examples, no jargon, and never mention that you are simplifying."
    )
}
 
# More info prompt
more_info_prompt = {
    "role": "system",
    "content": (
        "User would like more information on the topic just explained. Give new details and examples that build on what you "
        "already said, and remember to keep it simple enough for a 10 year old. "
        "Don't ask any questions at the end."
    )
}
 
buffer = 4

# If "messages" is not already in session state, we initialize it here with a message from assistant.
# Setting other state variables.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "awaiting_choice" not in st.session_state:
    st.session_state.awaiting_choice = False
if "pending" not in st.session_state:
    st.session_state.pending = None  # None, answer, or more_info

# For each message in st.session_state.messages, we display each message to the user.
chat_box = st.container(border=True, height=300)

with chat_box:
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        chat_message = st.chat_message(message["role"])
        chat_message.write(message["content"])
 
# Get user input.
# Assign the user's input to prompt.
prompt = st.chat_input(
    "Say 'hey' or ask a question.",
    disabled=st.session_state.awaiting_choice # Not currently awaiting Yes/No choice, so input field is not disabled.
)

# If user provides a prompt, append it to messages and we are now waiting for a regular answer.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending = "answer"
    st.rerun()

# If we are awaiting a Yes/No choice, show the buttons.
if st.session_state.awaiting_choice:
    with st.container(horizontal=True):
        # If user chooses Yes, append 'Yes' to messages, change awaiting choice back to False and we are now waiting for a "more_info" answer.
        if st.button("Yes", type="primary", key="yes_button"):
            st.session_state.messages.append({"role": "user", "content": "Yes"})
            st.session_state.awaiting_choice = False
            st.session_state.pending = "more_info"
            st.rerun()
        # If user chooses No, append 'No' to messages, change awaiting choice back to False and we go back to the initial 'help' message from the bot.
        if st.button("No", type="primary", key="no_button"):
            st.session_state.messages.append({"role": "user", "content": "No"})
            # Go back to asking what the bot can help with.
            st.session_state.messages.append(
                {"role": "assistant", "content": "What else can I help you with?"}
            )
            st.session_state.awaiting_choice = False
            st.rerun()
 
# Generate a response, either a normal answer or a "more info".
if st.session_state.pending:

    # Creating the prompt based on buffer and system prompt.
    # System prompt is pinned to the top.
    # The buffer is applied to messages after the system prompt.
    api_messages = (
        st.session_state.messages[:1] + st.session_state.messages[1:][-buffer:]
    )
 
    # On a "more info" answer, add the extra instruction at the end of the request.
    if st.session_state.pending == "more_info":
        api_messages = api_messages + [more_info_prompt]
 
    stream = st.session_state.client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=api_messages,
        stream=True,
    )

    with chat_box:
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
 
    # Store the answer and ask the follow-up question.
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append(
        {"role": "assistant", "content": "Do you want more info?"}
    )
 
    st.session_state.pending = None
    st.session_state.awaiting_choice = True
    st.rerun()