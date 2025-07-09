import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# .env 파일 로드
load_dotenv()

# Pinecone 인스턴스 생성
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 기존 인덱스 이름 사용 (미리 Pinecone 콘솔에서 만들어져 있어야 함)
index = pc.Index("text-to-sql-db")

# OpenAI 임베딩 모델 사용
embedding = OpenAIEmbeddings()

# LangChain vectorstore 객체 생성
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key="text"
)

# ✅ 테스트용 문서 삽입
docs = ["hello world", "how are you", "text to sql"]
vectorstore.add_texts(docs)
print("✅ 벡터 삽입 완료")

# 🔍 검색 테스트
query = "how can I write SQL?"
results = vectorstore.similarity_search(query)

print("🔍 검색 결과:")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")
