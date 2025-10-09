# ------------------- Imports -------------------
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# ------------------- CONFIG -------------------
API_KEY = "Your_API_Key"
File_Path = "test_doc.pdf"
prompt = "who plays golf"

# ------------------- Promt Template -------------------
prompt_template = PromptTemplate(template="""
                                 You are a helpful and creative AI assistant.
                                    Context:{context}
                                    Question: {input}
                                    Answer: 
                                            """,
                                    input_variables=["context", "input"]
                                            )

# ------------------- Initialize Model -------------------
client = ChatGoogleGenerativeAI(
                                model="gemini-2.5-flash",
                                google_api_key=API_KEY,
                                temperature=0.2,
                                max_output_tokens=100,
                                top_p=0.5,
                            )

# ------------------- Data Loader -------------------
data = loader = PyPDFLoader(File_Path).load()

# ------------------- Chunking -------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, length_function=len).split_documents(data)

# ------------------- Embedding -------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# ------------------- Storing in DB -------------------
vector_db = Chroma.from_documents(text_splitter, embedding_model)

# ------------------- retriever -------------------
retriever = vector_db.as_retriever(search_kwargs={"k":5}) if vector_db else None

# ------------------- Chaining -------------------
qa_chain = create_stuff_documents_chain(llm = client, prompt = prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ------------------- Invoke -------------------
response = rag_chain.invoke({"input": prompt})

# ------------------- Output -------------------
print("--->>>\n", 
      f"""Question: {response['input']}\n 
      Answer: {response['answer']}""")

