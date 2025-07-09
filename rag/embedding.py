import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# .env 로드
load_dotenv()

# SQLite DB 연결
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "test.db")

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

inspector = inspect(engine)
table_names = inspector.get_table_names()

# 문서화된 테이블 스키마 추출
docs = []
for table_name in table_names:
    columns = inspector.get_columns(table_name)
    column_strs = [f"- {col['name']} ({col['type']})" for col in columns]
    schema_text = f"Table: {table_name}\n" + "\n".join(column_strs)
    docs.append(Document(page_content=schema_text, metadata={"table": table_name}))

# Pinecone 초기화
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "text-to-sql-db"

# (선택) 인덱스가 없을 경우 생성
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # 또는 3072 → 아래 모델에 맞춰야 함
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 임베딩 모델 생성
embedding = OpenAIEmbeddings(model="text-embedding-3-small")  # → 차원: 1536

# 벡터스토어 연결
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding,
    text_key="text"
)

# Pinecone에 업로드
vectorstore.add_documents(docs)
print("✅ 스키마 벡터 업로드 완료")
