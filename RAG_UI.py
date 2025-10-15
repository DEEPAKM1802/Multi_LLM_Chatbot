#========================================================
# This is just an example json on how the data will be passed from backend
example_json = {
    "answer": "The capital of France is Paris.",
    "metrics": {
        "faithfulness": 0.95,
        "relevancy": 0.9,
        "correctness": 1.0
    }
}
#=======================================================




#========================== UI ==========================
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("⚙️ RAG AI Demo")

user_prompt = st.chat_input("Ask a question related to your document...", accept_file=True, file_type=["pdf"], width="stretch")

if user_prompt:
        st.chat_message("user").markdown(f"**Question:** {user_prompt['text']}")
        st.chat_message("assistant").markdown(f"**Answer:** {example_json['answer']}")
