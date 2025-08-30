from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Determine what the model is going to do, in that case, translation.
template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text from Portuguese to English:\n ```{initial_text}````"
)

# Determine what the model is going to do, in that case, summarization.
template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```"
)

# llm_en = ChatOpenAI(model="gpt-5-mini", temperature=0)
llm_en = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

# translated = translate.invoke({"initial_text": "A grama é verde."})
# print(translated)

# Execution order
# 1. Translate with template_translate with llm_en and use StrOutputParser to parse LLMResult in simple text
# 2. Summarize translated text with template_summary with llm_en and use StrOutputParser to parse LLMResult in simple text
result = pipeline.invoke({"initial_text": "LangChain é um framework para desenvolvimento de aplicações de IA"})
print(result)