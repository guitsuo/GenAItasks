

import numpy as np 
import pandas as pd 
import torch 


from huggingface_hub import notebook_login

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

# Enter your token ID (from huggingface)
token = ''

notebook_login()


# Check if GPU is set  
torch.cuda.is_available()


llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta", huggingfacehub_api_token = token)

chat_model = ChatHuggingFace(llm=llm)