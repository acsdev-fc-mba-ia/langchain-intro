from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(x:int) -> dict:
    return {"square_result": x * x}

template = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}. BE CONCISE!# "
)

# model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.5)

chain = square | template | model

# call the method square, pipe the return to the template that pipes the return to model, and gets an answer
result = chain.invoke(10)

print(result.content)