import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

load_dotenv()
llm=ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class QA(TypedDict):
    que:str
    ans:str

def llm_qa(state:QA) ->QA:
    question=state["que"]
    prompt=f"Answer the following question {question}"
    answer=llm.invoke(prompt).content
    state["ans"]=answer
    return state

graph=StateGraph(QA)
graph.add_node("llm_qa",llm_qa)
graph.add_edge(START,"llm_qa")
graph.add_edge("llm_qa",END)

res=graph.compile()
q={"que":"how far is moon from earth"}
response=res.invoke(q)
print(response["ans"])