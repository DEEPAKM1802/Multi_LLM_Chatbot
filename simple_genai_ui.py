import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# ------------------- Imports -------------------
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile

def rag_setup(api_key, model_name, chunk_size, chunk_overlap, top_k, temperature, max_output_tokens, top_p, file_path, prompt):
    # ------------------- Promt Template -------------------
    prompt_template = PromptTemplate(
        template="""
        You are a helpful and creative AI assistant.
        Context:{context}
        Question: {input}
        Answer:
        """,
        input_variables=["context", "input"]
    )

    # ------------------- Initialize Model -------------------
    client = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    # ------------------- Data Loader -------------------
    data = PyPDFLoader(file_path).load()

    # ------------------- Chunking -------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    ).split_documents(data)

    # ------------------- Embedding -------------------
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # ------------------- Storing in DB -------------------
    vector_db = Chroma.from_documents(text_splitter, embedding_model)

    # ------------------- retriever -------------------
    retriever = vector_db.as_retriever(search_kwargs={"k": 5}) if vector_db else None

    # ------------------- Chaining -------------------
    qa_chain = create_stuff_documents_chain(llm=client, prompt=prompt_template)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # ------------------- Invoke with Loading screen -------------------
    
    response = rag_chain.invoke({"input": prompt})

    # ------------------- RETRIEVED DOCS -------------------
    retrieved_docs = retriever.get_relevant_documents(prompt)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    references = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]

    # ======================================================
    #               METRICS CALCULATION SECTION
    # ======================================================

    # ---------- (1) Embedding Similarity ----------
    query_emb = embedding_model.embed_query(prompt)
    doc_embs = embedding_model.embed_documents(retrieved_texts)
    sims = cosine_similarity([query_emb], doc_embs)[0]
    embedding_metrics = {
        "avg_similarity": float(np.mean(sims)),
        "max_similarity": float(np.max(sims)),
        "min_similarity": float(np.min(sims)),
    }
    confidence_score = np.mean([
        cosine_similarity([query_emb], [doc_embedding])[0][0]
        for doc_embedding in doc_embs
    ])

    # ---------- (2) Basic Retrieval Metrics (IR-style) ----------
    threshold = np.mean(sims)
    relevant = sims > threshold
    k = len(sims)

    recall_at_k = np.sum(relevant) / len(sims) if len(sims) else 0
    precision_at_k = np.sum(relevant) / k if k else 0
    hit_rate_at_k = 1.0 if np.any(relevant) else 0.0
    if np.any(relevant):
        first_rel_rank = np.where(relevant)[0][0] + 1
        mrr = 1.0 / first_rel_rank
    else:
        mrr = 0.0

    retrieval_metrics = {
        "Recall@k": round(float(recall_at_k), 4),
        "Precision@k": round(float(precision_at_k), 4),
        "HitRate@k": round(float(hit_rate_at_k), 4),
        "MRR": round(float(mrr), 4),
    }
    retrieval_metrics["confidence_score"] = confidence_score

    # ---------- (3) Local Generation Evaluation (no RAGAS/OpenAI) ----------

    from sentence_transformers import SentenceTransformer, util

    # Reuse the same embedding model for semantic similarity
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    answer = response["answer"]
    st.write(answer)

    # Compute embeddings
    emb_contexts = sentence_model.encode(retrieved_texts, convert_to_tensor=True)
    emb_answer = sentence_model.encode(answer, convert_to_tensor=True)
    emb_query = sentence_model.encode(prompt, convert_to_tensor=True)

    # --- Faithfulness ---
    # Measures if answer is semantically close to retrieved context
    faithfulness_score = float(util.cos_sim(emb_answer, emb_contexts).max().item())

    # --- Context Relevance ---
    # Measures how related the retrieved context is to the query
    context_relevance = float(util.cos_sim(emb_query, emb_contexts).mean().item())

    # --- Answer Relevance ---
    # Measures if the answer is semantically close to the question
    answer_relevance = float(util.cos_sim(emb_query, emb_answer).item())

    # --- Conciseness / Fluency ---
    # Simple heuristic: penalize long or repetitive answers
    length_penalty = max(1.0, len(answer.split()) / 50)
    conciseness_fluency = round(1.0 / length_penalty, 4)

    # --- Factual Consistency ---
    # Local heuristic: how close answer & top-1 context are
    factual_consistency = float(util.cos_sim(emb_answer, emb_contexts[0]).item())

    generation_metrics = {
        "Faithfulness": round(faithfulness_score, 4),
        "Context Relevance": round(context_relevance, 4),
        "Answer Relevance": round(answer_relevance, 4),
        "Conciseness/Fluency": round(conciseness_fluency, 4),
        "Factual Consistency": round(factual_consistency, 4),
    }

    result = {
        "response": answer,
        "references": references,
        "embedding_metrics": embedding_metrics,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics
    }

    return result

def dispaly_results(result, user_prompt):
    st.write(result['response'])
    response = result["response"]
    references = result["references"]
    embedding_metrics = result["embedding_metrics"]
    retrieval_metrics = result["retrieval_metrics"]
    generation_metrics = result["generation_metrics"]

    st.chat_message("user").markdown(f"**Question:** {user_prompt}")
    with st.container():
        st.chat_message("assistant").markdown(f"**Answer:** {response}")
        pill_selection = st.pills("Meta Data", options=["Refrences", "Embedding Similarity", "Retrieval Metrics", "Generation Metrics"])
        if pill_selection == "Refrences":
            st.write(references)
        if pill_selection == "Embedding Similarity":
            for k, v in embedding_metrics.items():
                st.write(f"**{k}**: {v:.4f}")
        if pill_selection == "Retrieval Metrics":
            for k, v in retrieval_metrics.items():
                st.write(f"**{k}**: {v:.4f}")
        if pill_selection == "Generation Metrics":
            for k, v in generation_metrics.items():
                st.write(f"**{k}**: {v:.4f}")
 



def main():
    # -------------------- STREAMLIT UI --------------------
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

    # --- Sidebar: Configurable Parameters ---
    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", value="api_key_here")
    model_name = st.sidebar.selectbox("Model", ["gemini-2.5-flash", "gemini-2.0-pro"])

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    top_k = st.sidebar.slider("Top K Results", 1, 10, 5)
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.5, 0.1)
    
    chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 200, 50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 50, 10)
    max_output_tokens = st.sidebar.number_input("Max Output Tokens", 50, 500, 100)
    

    st.sidebar.markdown("---")

    # --- Main Chat Section ---
    st.title("💬 RAG Evaluation Chat")

    user_prompt = st.chat_input("Ask a question related to your document...", accept_file=True, file_type=["pdf"], width="stretch")
    # st.write(user_prompt)

    if user_prompt:
        if user_prompt['files']:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(user_prompt['files'][0].read())
                temp_path = tmp_file.name
                result = rag_setup(api_key, model_name, chunk_size, chunk_overlap, top_k, temperature, max_output_tokens, top_p, temp_path, user_prompt['text'])
                st.session_state["rag_result"] = result
                st.session_state["rag_prompt"] = user_prompt['text']
    if "rag_result" in st.session_state:
        dispaly_results(st.session_state["rag_result"], st.session_state["rag_prompt"])

if __name__ == "__main__":
    main()