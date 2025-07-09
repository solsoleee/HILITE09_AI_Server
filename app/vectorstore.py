import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("text-to-sql-db")

embedding = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")
retriever = vectorstore.as_retriever()
