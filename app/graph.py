from langgraph.graph import StateGraph, END
from app.agents import sql_agent_executor, general_agent
from app.prompts import supervisor_prompt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from app.database import get_schema, engine
from typing import List, TypedDict, Optional
from sqlalchemy import inspect
from app.vectorstore import retriever


class AgentState(TypedDict):
    history: List[BaseMessage]
    next_agent: Optional[str]
    relevant_tables: Optional[List[str]]

llm = ChatOpenAI(temperature=0, model="gpt-4")

# 25.7.7 추가 : 전체 입력 토큰을 절약하기 위해, 필요한 테이블을 유추해서 넘기는 별도의 노드 도입
table_inference_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in identifying relevant database tables from a user's natural language query. Given a list of available table names, identify the table names that are most likely needed to answer the query. Return only a comma-separated list of table names."),
    ("user", "Available tables:\n{table_names}\n\nUser query: {query}\nRelevant tables:")
])

def table_inference_node(state: AgentState):
    user_query = state["history"][-1].content
    
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    table_names_str = "\n".join(all_table_names)

    chain = table_inference_prompt | llm
    relevant_tables_str = chain.invoke({"table_names": table_names_str, "query": user_query}).content

    relevant_tables = [table.strip() for table in relevant_tables_str.split(',') if table.strip()]
    return {"relevant_tables": relevant_tables}

def supervisor_node(state: AgentState):
    prompt = supervisor_prompt.format(input=state["history"][-1].content)
    response = llm.invoke(prompt)
    if "SQL Agent" in response.content:
        return {"next_agent": "sql_agent", "history": state["history"]}
    else:
        return {"next_agent": "general_agent", "history": state["history"]}

def sql_agent_node(state: AgentState):
    user_input = state["history"][-1].content
    relevant_tables = state.get("relevant_tables", [])
    schema = get_schema(engine, tables=relevant_tables)
    context = state.get("retrieved_context", "")
    full_input = f"{context}\n\n{user_input}" if context else user_input

    response = sql_agent_executor.invoke({
        "input": full_input,
        "history": state["history"][:-1],
        "schema": schema
    })
    return {"history": state["history"] + [HumanMessage(content=response["output"])]}


def general_agent_node(state: AgentState):
    user_input = state["history"][-1].content
    response = general_agent(user_input)
    return {"history": state["history"] + [HumanMessage(content=response)]}

def retriever_node(state: AgentState):
    user_input = state["history"][-1].content
    docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "retrieved_context": context}


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("general_agent", general_agent_node)

workflow.add_node("retriever", retriever_node)

def route_supervisor(state: AgentState):
    if state["next_agent"] == "sql_agent":
        return "retriever"
    elif state["next_agent"] == "general_agent":
        return "general_agent"
    return END

workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "retriever": "retriever",
        "general_agent": "general_agent"
    }
)
workflow.add_node("table_inference", table_inference_node)
workflow.add_edge("retriever", "table_inference")
workflow.add_edge("table_inference", "sql_agent")

workflow.set_entry_point("supervisor")

graph = workflow.compile()
