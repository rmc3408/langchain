from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from duplicates import RedundantRetriever

from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI()
embeddings_model = OpenAIEmbeddings()
db = Chroma(persist_directory="db", embedding_function=embeddings_model)

retriever = db.as_retriever()
# retriever = RedundantRetriever(embeddings=embeddings_model, chroma=db) # clean duplicated result

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context: {context}
Question: {input}""")

document_chain = create_stuff_documents_chain(chat_model, prompt)

chain = create_retrieval_chain(retriever, document_chain)

res = chain.invoke({"input": "what is best english language phrase?"})
print(res)
