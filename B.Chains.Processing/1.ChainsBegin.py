from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!" 
)

# model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.5)

chain = question_template | model

result = chain.invoke({"name": "Allan"})
print(result.content)