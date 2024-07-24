from flask import Flask, request, render_template, session
import os
import textwrap
import glob
from pathlib import Path
from IPython.display import Markdown
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from llama_parse import LlamaParse
# from langchain_openai import OpenAI
from langchain.chains import ConversationChain

app = Flask(__name__)
app.secret_key = 'bubu'  # Set a secret key for session management


os.environ["GROQ_API_KEY"] = "my_groq_api"
os.environ["LLAMA_CLOUD_API_KEY"] = "myllamakey"

# markdown_directory = './data'

# all_docs = []

# markdown_files = glob.glob(os.path.join(markdown_directory, '*.md'))


# import asyncio
# import nest_asyncio
# nest_asyncio.apply()
# from httpx import ReadTimeout


# import tiktoken

# tokenizer = tiktoken.get_encoding('cl100k_base')

# # create the length function
# def tiktoken_len(text):
#     tokens = tokenizer.encode(
#         text,
#         disallowed_special=()
#     )
#     return len(tokens)


# # for 5 tries
# max_retries = 5
# retry_delay = 3  # in seconds


# async def process_md(markdown_file):
#     retries = 0
#     while retries < max_retries:
#         try:
#             print(f"Processing file: {markdown_file} (Attempt {retries + 1}/{max_retries})")

#             loader = UnstructuredMarkdownLoader(markdown_file)
#             loaded_documents = loader.load()
#             print(f"Loaded documents from markdown file: {markdown_file}")

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30, length_function=tiktoken_len, separators=[' ', ''])
#             docs = text_splitter.split_documents(loaded_documents)
#             print(f"Successfully split documents for file: {markdown_file}")

#             return docs

#         except ReadTimeout:
#             print(f"ReadTimeout occurred while processing {markdown_file}. Retrying {retries + 1}/{max_retries}...")
#             retries += 1
#             await asyncio.sleep(retry_delay)
#         except Exception as e:
#             print(f"An error occurred while processing {markdown_file}: {e}")
#             retries += 1
#             await asyncio.sleep(retry_delay)

#     print(f"Failed to process {markdown_file} after {max_retries} retries.")
#     return []

# if asyncio.get_event_loop().is_running():
#     tasks = [process_md(markdown_file) for markdown_file in markdown_files]
#     results = asyncio.gather(*tasks)
# else:
#     tasks = [process_md(markdown_file) for markdown_file in markdown_files]
#     results = asyncio.run(asyncio.gather(*tasks))

# for result in results:
#     all_docs.extend(result)


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# qdrant = Qdrant.from_documents(
# all_docs,
# embeddings,
# # location=":memory:",
# path="./db1",
# collection_name="document_embeddings",
# )

qdrant = Qdrant.from_existing_collection(
    embedding = embeddings,
    # location=":memory:",
    path="./db1",
    collection_name="document_embeddings",
    )

retriever = qdrant.as_retriever(search_kwargs={"k": 12})
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

llm = ChatGroq(temperature = 0.1, model_name="llama3-70b-8192")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

output_parser = StrOutputParser()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

instruction_to_system = """
formulate it as a detailed standalone question
DO not include the phrase IIT Mandi or "Indian Institute of Technology, Mandi"in your final output question, simply trim that part out of your question
Do NOT answer the question,
for example, If the question is "what is the design club of IIT mandi" your output should be "What is the design club, tell me about its activities." 
just reformulate it if needed and otherwise return it as is.
Focus on the subject of the question more than the object.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

question_chain = question_maker_prompt | llm | StrOutputParser()

qa_system_prompt = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that this is something I am not equipped to advice you on, don't try to make up an answer.

Context: {context}
If no data is available for a given question, say I do not have information about your question currently, for more details visit IIT Mandi official website.

Always try to give, to the point and relevant answers STRICTLY LESS THAN 500 WORDS EVERYTIME.

Try to give all the numbers, facts and figures mentioned in the supplied context but in a human readable, easy to read, PARAGRAPHICAL WELL FORMATTED LAYOUT.

REMEMBER DO NOT GIVE A PLACEHOLDER IF THE CONTEXT HAS NO LINK.

THERE SHOULD BE A CHRONOLOGY and a sense of continuity WITHIN THE RESPONSE. MOST RELEVANT POINTS SHOULD BE AT THE TOP.
Add personlization, give an answer as if you are directly talking to the person.

Try to answer in bullet points wherever necessary. If neat outputs with well defines paragraphs;
Keep the headings between *. example: *IIT Mandi*
"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    return question_chain
    
from langchain_core.runnables import RunnablePassthrough
retriever_chain = RunnablePassthrough.assign(
        context=contextualized_question | compression_retriever
    )

rag_chain = (
    retriever_chain
    | qa_prompt
    | llm
    | output_parser
)


print (" processed completely")

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/reply", methods=["GET", "POST"])
def chat():
    if "chat_history" not in session:
        session["chat_history"] = []

    msg = request.form["msg"]
    question = msg
    chat_history = session["chat_history"]

    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.clear
    chat_history.append(question)  # Storing the conversation as tuples

    session["chat_history"] = chat_history
    return ai_msg

if __name__ == "__main__":
    app.run()