from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# import os
# print(os.environ)

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
print(model)
message = model.invoke("Hello world!")

sprint(message.content)