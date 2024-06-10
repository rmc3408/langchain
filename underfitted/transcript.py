import tempfile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings()

# import whisper
# from pytube import YouTube

# YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=cdiD-9MMpb0"

# Let's do this only if we haven't created the transcription file yet.
# if not os.path.exists("transcription.txt"):
#     youtube = YouTube(YOUTUBE_VIDEO)
#     audio = youtube.streams.filter(only_audio=True).first()
#
#     # Let's load the base model. This is not the most accurate
#     # model but it's fast.
#     whisper_model = whisper.load_model("base")
#
#     with tempfile.TemporaryDirectory() as tmpdir:
#         file = audio.download(output_path=tmpdir)
#         transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()
#
#         with open("transcription.txt", "w") as file:
#             file.write(transcription)

with open("transcription.txt") as file:
    transcription = file.read()

# print(transcription[:100])

# embedded_query = embeddings.embed_query("Who is Mary's sister?")
# print(f"Embedding length: {len(embedded_query)}")
# print(embedded_query[:5])

loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(text_documents)
# print(documents[:1])

index_name = "youtube-transcript-rag-index"

pinecone = PineconeVectorStore.from_documents(
    documents, embeddings, index_name=index_name
)
print(pinecone.similarity_search("What is Hollywood going to start doing?")[:2])


