from fastapi import FastAPI
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

@app.post('/health')
def helth():
    return "Hello World"

# LLMs (Large Language Models)

@app.post('/llm-chat')
def llm_chat(query: str):
   llm = OpenAI()
   response = llm.invoke(query)
   return response


# Chat Models
@app.post('/openai-chat')
def openai_chat(query: str):
    chat = ChatOpenAI()
    messages = [
    SystemMessage(content="You are Ai assistant"),
    HumanMessage(content=query),
    ]
    response = chat.invoke(messages)
    
    return response.content


# Simple Prompts (PromptTemplate)
@app.post('/prompt-chat-PromptTemplate')
def prompt_chat_PromptTemplate():
    chat = ChatOpenAI()
    prompt_template = PromptTemplate.from_template(
        "Tell me an {adjective} explanation about {content}"
    )

    filled_prompt = prompt_template.format(
        adjective="interesting",
        content="LLM"
    )
    
    response = chat.invoke(filled_prompt)
    return response


# Prompt with Various Roles (ChatPromptTemplate)
@app.post('/prompt-chat-ChatPromptTemplate')
def prompt_chat_ChatPromptTemplate(query:str):
    chat = ChatOpenAI()
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello. how are you doing?"),
            ("ai", "I'm doing wll, thanks!"),
            ("human", "{user_input}"),
        ]
    )

    formatted_messages = chat_template.format_messages(
        name="Dissu",
        user_input=query
    )

    response = chat.invoke(formatted_messages)
    return response