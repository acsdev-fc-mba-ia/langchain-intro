from langchain.tools import tool
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result as a string."""
    try:
        # print("I am being used")
        result = eval(expression)  # cuidado: apenas para exemplo didático
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    print("I am being used")
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
        
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

tools = [calculator, web_search_mock]

prompt = hub.pull("hwchase17/react")
agent_chain = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True, 
    # max_iterations=5
)

print(agent_executor.invoke({"input": "What is the capital of Iran?"}))
# print(agent_executor.invoke({"input": "How much is 10 + 10?"}))