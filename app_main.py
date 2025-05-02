# app_main.py

import streamlit as st
from qa_chain import get_qa_chain
from doc_ingest import ingest_documents
from utils import save_chat_to_csv

from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Advanced Enterprise RAG Assistant", layout="wide")
st.title("📊 Enterprise Chat Assistant with RAG + Chat History + Multi-PDF")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Upload multiple PDFs
st.sidebar.header("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

# Main interaction
if uploaded_files:
    retriever = ingest_documents(uploaded_files)
    qa_chain = get_qa_chain(retriever)

    question = st.text_input("💬 Ask something from your documents")

    if question:
        with st.spinner("🔍 Generating answer..."):
            answer = qa_chain.run(question)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.success(answer)

    # Chat history
    st.subheader("🕓 Chat History")
    for item in st.session_state.chat_history[::-1]:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")

    # Download chat history
    st.download_button("⬇️ Download Q&A CSV", save_chat_to_csv(st.session_state.chat_history), "chat_history.csv")

else:
    st.info("Upload PDF files from the sidebar to begin.")