
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()


vectorstore1 = DocArrayInMemorySearch.from_texts(
    [
        "Mary's sister is Susana",
        "John and Tommy are brothers",
        "Patricia likes white cars",
        "Pedro's mother is a teacher",
        "Lucia drives an Audi",
        "Mary has two siblings",
    ],
    embedding=embeddings,
)

# print(vectorstore1.similarity_search_with_score(query="Who is Mary's sister?", k=3))

retriever1 = vectorstore1.as_retriever()
result1 = retriever1.invoke("Who is Mary's sister?")
print(result1)

retrieverChain = RunnableParallel(context=retriever1, question=RunnablePassthrough())
result2 = retrieverChain.invoke("What color is Patricia's car?")
print(result2)
