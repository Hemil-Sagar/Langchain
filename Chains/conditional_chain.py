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



load_dotenv()

# model = ChatOllama
# model = ChatOllama(
#     base_url="https://10a8706fb52c.ngrok-free.app",
#     model="qwen3:0.6b",
#     stream = False,
#     think = False
# )

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive','negative']= Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables= ['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )

branch_chain = RunnableBranch(
    (lambda x:x.sentiment== 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x:'Could not find sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is a great phone'})

print(result)

chain.get_graph().print_ascii()