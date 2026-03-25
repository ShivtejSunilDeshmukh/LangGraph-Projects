import os
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

class BMI(TypedDict):
    weight:float
    height:float
    bmi:float
    category:str

def calc_BMI(state : BMI) -> BMI:
    weight=state["weight"]
    height=state["height"]
    bmi=weight/(height**2)
    state["bmi"]=bmi
    return state

def label_BMI(state :BMI)->BMI:
    bmi=state["bmi"]
    if bmi<15:
        state["category"]="underweight"
    elif 15<bmi<25:
        state["category"]="normal"
    else:
        state["category"]="overweight"
    return state

graph=StateGraph(BMI)
graph.add_node("calc_BMI",calc_BMI)
graph.add_node("label_BMI",label_BMI)
graph.add_edge(START,"calc_BMI")
graph.add_edge("calc_BMI","label_BMI")
graph.add_edge("label_BMI",END)

response=graph.compile()

output=response.invoke({"weight":86,"height":1.8})
print(output)