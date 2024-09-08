import os
from pathlib import Path
import boto3

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.astradb import AstraDB
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# Load the environment variables
LOCAL_SECRETS = False

if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = st.secrets["ASTRA_VECTOR_ENDPOINT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "dabur"

os.environ["LANGCHAIN_PROJECT"] = "blueillusion"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Started")

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

#################
### Constants ###
#################

top_k_vectorstore = 8
top_k_memory = 3

###############
### Globals ###
###############

global embedding
global vectorstore
global retriever
global model
global chat_history
global memory

#############
### Login ###
#############
def check_password():
    def login_form():
        with st.form("credentials"):
            st.caption('Using a unique name will keep your content separate from other users.')
            st.text_input('Username', key='username')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        if len(st.session_state['username']) > 5:
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 Username must be 6 or more characters')
    return False

def logout():
    del st.session_state.password_correct
    del st.session_state.user
    del st.session_state.messages
    load_chat_history.clear()
    load_memory.clear()
    load_retriever.clear()

if not check_password():
    st.stop()

username = st.session_state.user

#######################
### Resources Cache ###
#######################

@st.cache_resource(show_spinner='Getting the Boto Session...')
def load_boto_client():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    return boto3.client("bedrock-runtime")

@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    return AstraDB(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )

@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="openai.gpt-4o-mini"):
    if 'openai' in model_id:
        gpt_version = 'gpt-4o-mini' if '4o' in model_id else 'gpt-3.5-turbo'
        return ChatOpenAI(
            temperature=0.2,
            model=gpt_version,
            streaming=True,
            verbose=True
        )
    return BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        streaming=True,
        model_kwargs={"temperature": 0.2},
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history(username):
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

@st.cache_data()
def load_prompt():
    template = """You're a helpful assistant tasked to help in recommending and helping them find Dabur products.
Prompt with some clarifying questions if you are not sure.
Use the following context to answer the question:
{context}
Use the previous chat history to answer the question:
{chat_history}
Question:
{question}
Answer in English."""
    return ChatPromptTemplate.from_messages([("system", template)])

#####################
### Session state ###
#####################

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi, I'm your personal shopping assistant!")]

# Add thumbs-up and thumbs-down feedback
def store_feedback(username, question, answer, feedback):
    """Store user feedback in the conversation history"""
    feedback_message = SystemMessage(content=f"User gave {feedback} feedback for this response: {answer}")
    chat_history.add_message(feedback_message)

############
### Main ###
############
st.set_page_config(initial_sidebar_state="collapsed")

# Apply CSS to position the logo in the top-right corner and reduce the size
st.markdown(
    """
    <style>
    .dabur-logo {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 50px;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the Dabur logo at the top-right corner of the main chat page
st.markdown('<img src="./public/dabur_logo.png" class="dabur-logo">', unsafe_allow_html=True)

with st.sidebar:
    # Display the Dabur logo in the sidebar
    st.image('./public/dabur_logo.png', use_column_width=True)
    
    st.button(f"Logout '{username}'", on_click=logout)
    bedrock_runtime = load_boto_client()
    embedding = load_embedding()
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    chat_history = load_chat_history(username)
    memory = load_memory()
    prompt = load_prompt()

with st.sidebar:
    with st.form('delete_memory'):
        st.caption('Delete the history in the conversational memory.')
        submitted = st.form_submit_button('Delete chat history')
        if submitted:
            memory.clear()

with st.sidebar:
    st.caption('Choose the LLM model')
    model_id = st.selectbox('Model', [
        'openai.gpt-4o-mini',
        'openai.gpt-3.5',
        'meta.llama2-70b-chat-v1',
        'meta.llama2-13b-chat-v1',
        'amazon.titan-text-express-v1',
    ])
    model = load_model(model_id)

st.markdown("<style> img {width: 200px;} </style>", unsafe_allow_html=True)

# Draw all messages and allow feedback
for i, message in enumerate(st.session_state.messages):
    st.chat_message(message.type).markdown(message.content)
    if message.type == "assistant":
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", key=f"thumbs_up_{i+1}"):  # Emoji for thumbs up
                store_feedback(username, st.session_state.messages[i-1].content, message.content, "positive")
       
        with col2:
            if st.button("👎", key=f"thumbs_down_{i+1}"):  # Emoji for thumbs down
                store_feedback(username, st.session_state.messages[i-1].content, message.content, "negative")

# Get a prompt from the user
if question := st.chat_input("How can I help you?"):
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        history = memory.load_memory_variables({})

        # Ensure 'chat_history' is in the correct format
        if 'chat_history' not in history:
            history['chat_history'] = ""

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: history['chat_history'],  # Pass chat history here
            'question': lambda x: x['question']
        })

        chain = inputs | prompt | model
        response = chain.invoke({'question': question, 'chat_history': history['chat_history']}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        content = response.content

        response_placeholder.markdown(content)
        memory.save_context({'question': question}, {'answer': content})
        st.session_state.messages.append(AIMessage(content=content))

        # Add thumbs up and thumbs down buttons for the new response
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}"):
                store_feedback(username, question, content, "positive")
        with col2:
            if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}"):
                store_feedback(username, question, content, "negative")
                         