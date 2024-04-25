from fastapi import FastAPI
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage


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