from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from app.database import engine, get_schema
from app.prompts import sql_agent_prompt, general_agent_prompt

import os

llm = ChatOpenAI(temperature=0, model="gpt-4")

db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


db_type = os.getenv("DB_TYPE", "mysql")
sql_prompt = sql_agent_prompt.partial(db_type=db_type)

sql_agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    agent_type="openai-tools",
    prompt=sql_prompt
)

def general_agent(input_text):
    prompt = general_agent_prompt.format(input=input_text)
    response = llm.invoke(prompt)
    return response.content