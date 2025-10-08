import streamlit as st
import os, tempfile, json, re, hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Persistent storage in session
if "file_cache" not in st.session_state:
    st.session_state["file_cache"] = {}

if "global_vectordb" not in st.session_state:
    st.session_state["global_vectordb"] = None

# ------------------ Data Processing ------------------
class DataProcessing:
    @staticmethod
    def document_loader(file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        loaded_document = loader.load()
        os.remove(tmp_path)
        return loaded_document

    @staticmethod
    def text_splitter(data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        return text_splitter.split_documents(data)

    @staticmethod
    def vector_database(chunks):
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        return Chroma.from_documents(chunks, embedding_model, persist_directory="./db")

    def get_processed_data(self, file):
        loaded_data = self.document_loader(file)
        tokenized_data = self.text_splitter(loaded_data)
        return self.vector_database(tokenized_data)


# ------------------ Google GenAI Adapter ------------------
class GoogleGenAIAdapter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None

    def initialize(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
            max_output_tokens=4096,
            top_p=0.5,
        )

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
                        You are a helpful and creative AI assistant.

                        {system_prompt}

                        Context:
                        {context}

                        Question:
                        {input}

                        Answer:
                        """,
            input_variables=["system_prompt", "context", "input"],
        )

    def respond(self, user_prompt: str, file=None):
        system_prompt = system_prompt = """
                        You are an assistant analyzing support chat messages.
                        If no context document is uploaded, ask the user to upload one.
                        If the answer is not in the provided context, clearly say: "Answer not found in the database."
                        Do not guess or infer information not contained in the document.
                        """

        if self.model is None:
            self.initialize()

        # Load from cache or process
        retriever = None
        # ---------------- Handle File Uploads + Persistent Cache ----------------
        if file:
            file_id = hashlib.md5((file.name + str(file.size)).encode()).hexdigest()
            if file_id not in st.session_state["file_cache"]:
                vectordb = DataProcessing().get_processed_data(file)
                st.session_state["file_cache"][file_id] = vectordb

                # If global vectordb doesn't exist, set it
                if st.session_state["global_vectordb"] is None:
                    st.session_state["global_vectordb"] = vectordb
                else:
                    # Add new document chunks to global vector DB
                    st.session_state["global_vectordb"].add_documents(vectordb.get()['documents'])
                    st.session_state["global_vectordb"].persist()
        else:
            # If no file is uploaded but we have global_vectordb, reuse it
            if st.session_state["global_vectordb"]:
                vectordb = st.session_state["global_vectordb"]
            else:
                vectordb = None

        retriever = vectordb.as_retriever(search_kwargs={"k":5}) if vectordb else None


        # Memory persisted via Streamlit session state
        if "chat_memory" not in st.session_state:
            st.session_state["chat_memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory = st.session_state["chat_memory"]

        # Build prompt + RAG chain
        prompt = self.get_prompt_template().partial(system_prompt=system_prompt)
        qa_chain = create_stuff_documents_chain(self.model, prompt)

        if retriever:
            rag_chain = create_retrieval_chain(retriever, qa_chain)
            response = rag_chain.invoke({"input": user_prompt})

            answer = response.get("answer", response.get("output_text", "No response"))

            # Extract sources (if available)
            sources = []
            docs = response.get("context", []) or response.get("source_documents", [])
            for d in docs:
                metadata = d.metadata if hasattr(d, "metadata") else {}
                snippet = d.page_content[:200].replace("\n", " ") + "..."
                source_info = f"📄 **Source:** {metadata.get('source', 'Unknown')} — \"{snippet}\""
                sources.append(source_info)

            if sources:
                answer += "\n\n---\n**Retrieved from:**\n" + "\n".join(sources)

        else:
            result = self.model.invoke(user_prompt)
            answer = getattr(result, "content", str(result))

        # Save chat memory context
        memory.save_context({"user": user_prompt}, {"assistant": answer})
        return answer


# ------------------ Streamlit UI ------------------
def main():
    st.set_page_config(page_title="C.R.A.G", layout="wide")
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        st.title(":brain: C.R.A.G")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input(
        "Say something and/or attach a PDF",
        accept_file=True,
        file_type=["pdf"],
    )

    if prompt:
        text = prompt["text"]
        file = prompt["files"][0] if prompt["files"] else None

        if not text:
            st.warning("Please enter a question.")
            return

        # Display user message
        st.chat_message("user").markdown(text)
        st.session_state["messages"].append({"role": "user", "content": text})

        # Get response
        result = GoogleGenAIAdapter("xxxx").respond(user_prompt=text, file=file)

        # Display AI message
        st.chat_message("assistant").markdown(result)
        st.session_state["messages"].append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()
