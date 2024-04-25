from fastapi import FastAPI
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator


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


# PydanticOutputParser (This metdos just work with openai==0.28.0, not with newest versions)
@app.post('/OutputParser')
def OutputParser(query: str):
    model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    class Joke(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

        @validator("setup")
        def question_ends_with_question_mark(cls, field):
            if field[-1] != "?":
                raise ValueError("Badly formed question!")
            return field

    # Set up a PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=Joke)


    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Combine prompt, model, and parser to get structured output
    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": query})

    # Parse the output using the parser
    parsed_result = parser.invoke(output)

    return parsed_result