import os
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

from sqlalchemy.engine import Engine

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    DB_USER = os.getenv("DB_USER_READONLY", "your_readonly_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD_READONLY", "your_readonly_password")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME")
    DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

# 로컬 환경에서는 테스트 DB에 연결
else:
    DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Schema 추출 기능
def get_schema(engine: Engine, tables: list[str] = None) -> str:
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()

    if tables:
        table_names_to_get = [t for t in tables if t in all_table_names]
    else:
        table_names_to_get = all_table_names

    schema_str = ""
    for table_name in table_names_to_get:
        columns = inspector.get_columns(table_name)
        schema_str += f"Table: {table_name}\n"
        for col in columns:
            schema_str += f"  {col['name']} {str(col['type'])}\n"
        schema_str += "\n"
    return schema_str