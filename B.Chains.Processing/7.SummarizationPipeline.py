# from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()
long_text = """Dawn threads a pale gold through the alley of glass.
The city yawns in a chorus of brakes and distant sirens.
Windows blink awake, one by one, like sleepy eyes.
Streetcloth of steam curls from manholes, a quiet river.
Coffee steam spirals above a newspaper's pale print.
Pedestrians sketch light on sidewalks, hurried, loud with umbrellas.
Buses swallow the morning with their loud yawns.
A sparrow perches on a steel beam, surveying the grid.
The subway sighs somewhere underground, a heartbeat rising.
Neon still glows in the corners where night refused to retire.
A cyclist cuts through the chorus, bright with chrome and momentum.
The city clears its throat, the air turning a little less electric.
Shoes hiss on concrete, a thousand small verbs of arriving.
Dawn keeps its promises in the quiet rhythm of a waking metropolis.
The morning light cascades through towering windows of steel and glass,
casting geometric shadows on busy streets below.
Traffic flows like rivers of metal and light,
while pedestrians weave through crosswalks with purpose.
Coffee shops exhale warmth and the aroma of fresh bread,
as commuters clutch their cups like talismans against the cold.
Street vendors call out in a symphony of languages,
their voices mixing with the distant hum of construction.
Pigeons dance between the feet of hurried workers,
finding crumbs of breakfast pastries on concrete sidewalks.
The city breathes in rhythm with a million heartbeats,
each person carrying dreams and deadlines in equal measure.
Skyscrapers reach toward clouds that drift like cotton,
while far below, subway trains rumble through tunnels.
This urban orchestra plays from dawn until dusk,
a endless song of ambition, struggle, and hope."""

spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
parts = spliter.create_documents([long_text])

# print all parts
# for part in parts:
#     print(part.page_content)
#     print("-" * 10)

# llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0)

# LCEL map stage: summarize each chunk
map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n{context}")
map_chain = map_prompt | llm | StrOutputParser()

def prepare_map_inputs_function(docs):
    return [{"context": d.page_content} for d in docs]

# Replace the lambda with the function
prepare_map_inputs = RunnableLambda(prepare_map_inputs_function)
map_stage = prepare_map_inputs | map_chain.map()

# returns a array where each element is a summary of the corresponding chunk
# print(map_stage.invoke(parts))
# print("=====================")

# LCEL reduce stage: combine summaries into one final summary
reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n{context}")
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_input = RunnableLambda(lambda summaries: {"context": "\n".join(summaries)})
pipeline = map_stage | prepare_reduce_input | reduce_chain

# Order of execution
#
# 1. Input Preparation
#    The text is split into smaller chunks (parts) using RecursiveCharacterTextSplitter.
# 2. Map Stage: Summarizing Each Chunk
#    Each chunk is formatted into a dictionary (prepare_map_inputs).
#    The language model generates summaries for each chunk using a prompt (map_chain).
# 3. Prepare Reduce Input
#    The output of the map stage is an array of summaries, one for each chunk.
#    These summaries are combined into a single input for the reduce stage (prepare_reduce_input).
# 4. Reduce Stage: Combining Summaries
#    Individual summaries are combined into a single string (prepare_reduce_input).
#    The language model generates a final concise summary from the combined input (reduce_chain).
# 5. Pipeline Execution
#    The pipeline orchestrates the map and reduce stages, processing the chunks and combining results.
# 6. Output
#    The final summary is stored in result.
result = pipeline.invoke(parts)
print(result)
