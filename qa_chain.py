from langchain.chains import RetrievalQA 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    chain = RetrievalQA.from_chain_type(llm = llm , retriever = retriever, chain_type="stuff")
    return chain