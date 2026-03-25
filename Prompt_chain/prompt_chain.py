import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

load_dotenv()

llm=ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="openrouter/free"
)

class Blog_state(TypedDict):
    title:str
    outline:str
    content:str


def outline_state(state : Blog_state) -> Blog_state:
    title=state["title"]
    prompt=f"generate a detailed outline for blog on topic {title}"
    state["outline"]=llm.invoke(prompt).content
    return state
    
def content_state(state : Blog_state) -> Blog_state:
    title=state["title"]
    outline=state["outline"]
    prompt=f"write a detailed blog  on {title}. use this outline {outline}"
    state["content"]=llm.invoke(prompt).content
    return state

graph=StateGraph(Blog_state)
graph.add_node("outline_state",outline_state)
graph.add_node("content_state",content_state)

graph.add_edge(START,"outline_state")
graph.add_edge("outline_state","content_state")
graph.add_edge("content_state",END)

result=graph.compile()
input={"title":"Rise of AI in India"}
response=result.invoke(input)
print(response)