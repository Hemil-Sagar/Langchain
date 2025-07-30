import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
os.environ["GOOGLE_API_KEY"] = "AIzaSyDdApDJ-DDgIP6w4Vr5iQjMC4BUrTFKGvo"  # Replace with your actual Google API key

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and perform web searches."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

question = "What is the current price of gold in ahmedabvad, and are there any local news events affecting ot today?"
print(f"Asking Question: {question}\n")

response = agent_executor.invoke({"input": question})

print("\n --- Final Answer ---")
print(response["output"])