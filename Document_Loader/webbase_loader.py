from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
prompt = PromptTemplate(
    template='Answer the following question \n {question} in one line from the following text- \n {text}',
    input_variables=['question','text']
)
url = 'https://medium.com/@dmohankrishna99/understanding-open-voice-part-1-of-voice-cloning-099b5bf754b0'

parser = StrOutputParser()

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the product that we are talking about?','text':docs[0].page_content}))


