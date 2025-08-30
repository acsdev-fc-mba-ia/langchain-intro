from langchain_core.runnables import RunnableLambda

def parse_number(text:str) -> int:
    return int(text.strip())

# Wraps a function in something that can be passed to a pipe
parse_runnable = RunnableLambda(parse_number)

number = parse_runnable.invoke("10")
print(f"Number: {number}, Type: {type(number)}")