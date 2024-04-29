from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

embeddings_model = OpenAIEmbeddings()

raw_documents = TextLoader("facts.txt").load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20
)

documents = text_splitter.split_documents(raw_documents)

db = Chroma.from_documents(documents, embeddings_model, persist_directory='db')

# db.similarity_search("what is best english language phrase?")
