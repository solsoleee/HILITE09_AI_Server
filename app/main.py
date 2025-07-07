from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import graph
from langchain_core.messages import HumanMessage

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(request: QueryRequest):
    initial_state = {"history": [HumanMessage(content=request.query)]}
    result = graph.invoke(initial_state)
    return {"response": result["history"][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)