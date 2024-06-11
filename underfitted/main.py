import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from transcript import pinecone

load_dotenv()


template = """
Answer the question based on the context below. If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings()
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)

chain1 = prompt | model | parser
result1 = chain1.invoke({
    "context": "Mary's sister is Susana",
    "question": "Who is Mary's sister?"
})
print(result1)


translation_prompt = ChatPromptTemplate.from_template("Translate {answer} to {language}")
translation_chain = {"answer": chain1, "language": itemgetter("language")}

chain2 = translation_chain | translation_prompt | model | parser

result2 = chain2.invoke(
    {
        "context": "Mary's sister is Susana. She doesn't have any more siblings.",
        "question": "How many sisters does Mary have?",
        "language": "Portuguese Brazilian",
    }
)

print(result2)

# TRANSCRIPT
# with open("transcription.txt") as file:
#     transcription = file.read()
#
# print(transcription[:100])

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

#retriever1 = vectorstore1.as_retriever()
retriever1 = pinecone.as_retriever()

retrieverChain = RunnableParallel(context=retriever1, question=RunnablePassthrough())

chain3 = retrieverChain | prompt | model | parser
# result3 = chain3.invoke("What color is Patricia's car?")
# print(result3)
#
# result4 = chain3.invoke("What car does Lucia drive?")
# print(result4)

result5 = chain3.invoke("What is the video about in 20 words.")
print(result5)
