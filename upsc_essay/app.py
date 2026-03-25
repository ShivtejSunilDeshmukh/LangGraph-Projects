import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
import operator

load_dotenv()

st.set_page_config(
    page_title="UPSC Essay Evaluator",
    page_icon="📝",
    layout="wide"
)

# ---------------- LLM ---------------- #

llm=ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openrouter/free"
)

class initial(BaseModel):
    feedback:str=Field(description="give a proper feedback based on language quality")
    score:int=Field(description="Assign a score out of 10",ge=0,le=10)

model=llm.with_structured_output(initial)

# ---------------- STATE ---------------- #

class upsc(TypedDict):
    essay:str
    lang_fb:str
    analysis_fb:str
    clarity_fb:str
    overall_fb:str
    score:Annotated[list[int],operator.add]
    final_score: float

# ---------------- EVALUATORS ---------------- #

def eval_lang(state:upsc):
    prompt=f"evaluate the given essay and provide a language feedback and assign a score out of 10 {state['essay']} "
    ans=model.invoke(prompt)
    return {"lang_fb":ans.feedback,"score":[ans.score]}

def eval_anlysis(state:upsc):
    prompt=f"evaluate the depth of analysis essay and provide a feedback and assign a score out of 10 {state['essay']} "
    ans=model.invoke(prompt)
    return {"analysis_fb":ans.feedback,"score":[ans.score]}

def eval_thought(state:upsc):
    prompt=f"evaluate the clarity of thoughts of essay and provide a feedback and assign a score out of 10 {state['essay']} "
    ans=model.invoke(prompt)
    return {"clarity_fb":ans.feedback,"score":[ans.score]}

def final_eval(state:upsc):
    prompt=f"evaluate the overall essay using given language ,depth,and clarity of thought of feedback {state['lang_fb']},{state['analysis_fb']},{state['clarity_fb']} "
    ans=llm.invoke(prompt)
    avg=sum(state["score"])/len(state["score"])
    return {"overall_fb":ans.content,"final_score":avg}

# ---------------- GRAPH ---------------- #

graph=StateGraph(upsc)

graph.add_node("eval_lang",eval_lang)
graph.add_node("eval_anlysis",eval_anlysis)
graph.add_node("eval_thought",eval_thought)
graph.add_node("final_eval",final_eval)

graph.add_edge(START,"eval_lang")
graph.add_edge(START,"eval_anlysis")
graph.add_edge(START,"eval_thought")

graph.add_edge("eval_lang","final_eval")
graph.add_edge("eval_anlysis","final_eval")
graph.add_edge("eval_thought","final_eval")

graph.add_edge("final_eval",END)

result=graph.compile()

# ---------------- UI ---------------- #

st.title("📝 UPSC Essay AI Evaluator")
st.markdown("Evaluate your essay based on **Language, Analysis, and Clarity of Thought**.")

st.divider()

essay_input = st.text_area(
    "✍️ Paste your Essay Below",
    height=300,
    placeholder="Write or paste your UPSC essay here..."
)

col1, col2 = st.columns([1,4])

with col1:
    evaluate = st.button("🚀 Evaluate Essay")

with col2:
    st.caption("AI will analyze your essay using multiple evaluators.")

# ---------------- RUN EVALUATION ---------------- #

if evaluate:

    if essay_input.strip()=="":
        st.warning("⚠️ Please enter an essay first.")
    else:

        with st.spinner("🔍 Evaluating your essay..."):
            response=result.invoke({"essay":essay_input})

        st.divider()

        st.subheader("📊 Final Score")

        st.metric(
            label="Overall Score",
            value=f"{response['final_score']:.2f}/10"
        )

        st.progress(response["final_score"]/10)

        st.divider()

        # Scores
        s1, s2, s3 = response["score"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Language", f"{s1}/10")
            st.progress(s1/10)

        with col2:
            st.metric("Analysis", f"{s2}/10")
            st.progress(s2/10)

        with col3:
            st.metric("Clarity", f"{s3}/10")
            st.progress(s3/10)

        st.divider()

        with st.expander("🧠 Language Feedback", expanded=True):
            st.write(response["lang_fb"])

        with st.expander("📚 Analysis Feedback"):
            st.write(response["analysis_fb"])

        with st.expander("💡 Clarity Feedback"):
            st.write(response["clarity_fb"])

        with st.expander("⭐ Overall Evaluation"):
            st.write(response["overall_fb"])