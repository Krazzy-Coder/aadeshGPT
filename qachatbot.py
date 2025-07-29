import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import datetime
import os

load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD")
os.environ["WATCHDOG_OBSERVER_TIMEOUT"] = "1.0"

st.set_page_config(page_title="Ask My Resume", page_icon="📄")
st.title("📄 Talk About my Professional Career with my AI-powered Personal Chat Bot 🤖")
password = st.text_input("Enter password to access the bot", type="password")

if password != APP_PASSWORD:
    st.warning("🔐 This bot is private. Please enter the correct password.")
    st.stop()

with st.sidebar:
    st.image("me.jpg", width=200)
    st.markdown("""
    ### Aadesh Srivastava
    **DSA Master | Full-Stack Engineer | AI Explorer**  
    📍 Bangalore, India  
    ✉️ aadeshsrivastava48@gmail.com  
    📞 +918795969377  
    🧠 RAG Groq-powered career Q&A chatbot
    """)
    st.markdown("---")
    st.markdown("### 👀 Check out my profiles")

    st.markdown(
    """
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srivastava-aadesh/)  

    [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/Krazzy-Coder/)    
    
    [![LeetCode](https://img.shields.io/badge/LeetCode-orange?logo=leetcode&logoColor=white)](https://leetcode.com/u/aadeshsrivastava48/)   

    ![LeetCode Stats](https://leetcard.jacoblin.cool/aadeshsrivastava48?ext=heatmap)
    """,
    unsafe_allow_html=True
    )
    st.markdown("💡 *Ask about my experience, skills, tools I have used, or projects I have worked on!*")


@st.cache_resource
def setup_chain():

    loader = PyPDFLoader("./files/resume.pdf")
    pdf_documents = loader.load()

    text_loader = TextLoader("./files/info.txt")
    text_documents = text_loader.load()

    all_documents = pdf_documents + text_documents

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = splitter.split_documents(all_documents)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()


    llmGroq = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_retries=2,
    )


    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant who answers questions based on a person's resume.
    Answer based on the context only. If you don't know, say so.
    Think step by step before providing a detailed answer.

    <context>
    {context}
    </context>
    Question: {input}
    """)


    document_chain = create_stuff_documents_chain(llmGroq, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain

rag_chain = setup_chain()


query = st.text_input("Ask a question about my experience, skills, or background:")
st.markdown("---")
st.markdown("#### 💡 Need ideas? Try asking me things like:")
st.markdown("""
- 👩‍💻 What backend frameworks does Aadesh have experience with?
- 🧠 Has he worked on any AI or machine learning projects?
- 🌐 Does he know React or other frontend technologies?
- 🏢 What kind of companies has he worked with?
- 📚 What are his educational qualifications?
- 🔧 What tools and platforms is he skilled in?
""")
if query:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": query})
        with open("question_log.txt", "a") as log_file:
            log_file.write(f"[{datetime.datetime.now()}] Question: {query}\n")
            log_file.write(f"Response: {response['answer']}\n\n")
        st.markdown("### 🧠 Answer")
        st.write(response["answer"])
