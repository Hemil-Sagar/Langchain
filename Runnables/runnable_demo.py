from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
