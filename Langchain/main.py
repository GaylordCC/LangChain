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
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import DatetimeOutputParser
from langchain.chains import LLMChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter




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
def prompt_chat_PromptTemplate(query: str):
    chat = ChatOpenAI()
    prompt_template = PromptTemplate.from_template(
        "Tell me an {adjective} explanation about {content}"
    )

    filled_prompt = prompt_template.format(
        adjective="interesting",
        content=query
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
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ]
    )

    formatted_messages = chat_template.format_messages(
        name="Dissu",
        user_input=query
    )

    response = chat.invoke(formatted_messages)
    return response


# Pydantic Output Parser (This metdos just work with openai==0.28.0, not with newest versions)
@app.post('/OutputParser')
def OutputParser(query: str):
    model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    class Joke(BaseModel):
        setup: str = Field(description="question to set up an doubt")
        punchline: str = Field(description="answer to resolve the doubt")

        @validator("setup")
        def question_ends_with_question_mark(cls, field):
            if field[-1] != "?":
                raise ValueError("Badly formed question!")
            return field

    # Set up a Pydantic Output Parser
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


# Simple Json Output Parser
@app.post('/SimpleJson')
def simple_json_output_parser(query: str):
    model = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

    # Create a JSON prompt
    json_prompt = PromptTemplate.from_template(
        "Return a JSON with 'birthday'and 'birthplace' that answer the following question: {question}"
    )

    # Initialize the JSON parser
    json_parser = SimpleJsonOutputParser()

    # Create a chain with the prompt, model, and parser
    json_chain = json_prompt | model | json_parser

    # Stream through the results
    result_list = list(json_chain.stream({"question": query}))


    return result_list


# Comma Separated List Output Parser
@app.post('/CommaSeparated')
def comma_separated(query: str):
    model = OpenAI()

    # Initialize the parser
    output_parser = CommaSeparatedListOutputParser()

    # Create format instructions
    format_instructions = output_parser.get_format_instructions()

    # Create a prompt to request a list
    prompt = PromptTemplate(
        template="List five {subject}. \n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Generate the output
    output =model(prompt.format(subject=query))

    # Parse the output using the parser
    parsed_result = output_parser.parse(output)

    return parsed_result

# Datetime Output Parser
@app.post('/')
def datetime_output_parser(query: str):
    # Initialize the DatetimeOutputParser
    output_parser = DatetimeOutputParser()

    # Create a prompt with format instructions
    template = """
    Answer the user's question:
    {question}
    {format_intructions}
    """
    prompt = PromptTemplate.from_template(
        template,
        partial_variables={"format_intructions": output_parser.get_format_instructions()}, 
    )

    # Create a chain with the prompt and language model
    chain = LLMChain(prompt=prompt, llm=OpenAI())

    # Run the chain
    output = chain.run(query)

    # Parse the output using the datetime parser
    parsed_result = output_parser.parse(output)

    return parsed_result


# Document Ingest
@app.post('/document-ingest-langchain')
def document_ingest_langchain():
    loader = CSVLoader(
        file_path='./keywords_rotobot.csv',
        csv_args={
            'delimiter': ';',
            'quotechar': '"',
            'fieldnames': ["keyword", "metadata"]
    })
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    docs = text_splitter.split_documents(documents)

    return docs