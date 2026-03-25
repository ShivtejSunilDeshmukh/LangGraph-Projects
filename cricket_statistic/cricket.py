import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm=ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openrouter/free"
)

class B_state(TypedDict):
    runs:int
    four:int
    six:int
    balls:int

    sr:float
    bpb:float
    b_per:float
    summary:str

def calc_sr(state:B_state):
    sr=(state["runs"]/state["balls"])*100
    return {"sr":sr}

def calc_bpb(state:B_state):
    bpb=(state["balls"]/(state["four"]+state["six"]))
    return {"bpb":bpb}

def calc_bper(state:B_state):
    b_per=((state["four"]*4+state["six"]*6)/state["runs"])*100
    return {"b_per":b_per}

def calc_summary(state:B_state):
    summary=f"""
    Strike Rate : {state['sr']}, 
    Balls Per Boundary :{state['bpb']},
    Boundary Percentage : {state['b_per']}

    
    """
    return {"summary":summary}

graph=StateGraph(B_state)

graph.add_node("calc_sr",calc_sr)
graph.add_node("calc_bpb",calc_bpb)
graph.add_node("calc_bper",calc_bper)
graph.add_node("calc_summary",calc_summary)


graph.add_edge(START,"calc_sr")
graph.add_edge(START,"calc_bpb")
graph.add_edge(START,"calc_bper")

graph.add_edge("calc_sr","calc_summary")
graph.add_edge("calc_bpb","calc_summary")
graph.add_edge("calc_bper","calc_summary")

graph.add_edge("calc_summary",END)

result=graph.compile()
ab={"runs":95,"four":5,"six":6,"balls":49}
response=result.invoke(ab)
print(response["summary"])