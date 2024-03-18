

import numpy as np 
import pandas as pd 
import torch 
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper

from huggingface_hub import notebook_login

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

# Enter your token ID (from huggingface)
token = ''

notebook_login()


# Check if GPU is set  
torch.cuda.is_available()

# Open source model 
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token = token)

chat_model = ChatHuggingFace(llm=llm)

# Load tools 
tools = load_tools(["serpapi", 'llm-math'], serpapi_api_key = serpapi, llm=llm) #,  # , "llm-math"

#setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors = True)

agent_executor.invoke(
    { #holder of the speed skating world record on 500 meters 
        "input": "Quanto custa criar uma conta com o banco BV? Adicione uma taxa de 10 reais"
    
    }
    
)