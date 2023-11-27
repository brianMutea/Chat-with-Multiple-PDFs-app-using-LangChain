import streamlit as st
import pypdf
import tiktoken
from pypdf import PdfReader
import pdf2image
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, UnstructuredFileLoader, DocugamiLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain

from transformers import AutoModelForCausalLM  # for DeciLM-6B model

# load the environments
from dotenv import load_dotenv
load_dotenv()

# set some variables

CHUNK_SIZE = 1000

# # Using HuggingFaceEmbeddings with the chosen embedding model
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# jxm/sentence-transformers_all-MiniLM-L6-v2__msmarco__128

# LLM
# model_id = "Deci/DeciLM-6b"
# llm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model_id = "google/flan-t5-base"
llm = HuggingFaceHub(
    repo_id=model_id, model_kwargs={"temperature": 0.5, "max_length": 64})


def creat_vector_db(uploaded_pdfs) -> FAISS:
    """"""

    pdf_texts = ""

    # read the file first

    for pdf in uploaded_pdfs:

        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_texts += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=100)

    texts = text_splitter.split_text(pdf_texts)

    # create vector db for similarity search
    vector_db = FAISS.from_texts(texts, EMBEDDINGS)

    return vector_db


def define_custom_prompt():
    custom_prompt_template = """You have been given the following documents to answer the user's question.
    If you do not have information from the information given to answer the questions just say 'I don't know the answer" and don't try to make up an answer.
    Context: {context}
    Question: {question}
    Give a detailed helpful answer and nothing more.
    Helpful answer:
"""
    prompt = PromptTemplate(template="", input_variables=[
                            "context", "question"])

    return prompt


def conversation_chain(vector_db):
    """Chain to retrieve answers. the chain takes  the documents and
    makes a call to the DeciLM-6b llm """

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
    )

    return conversation_chain


# Create a Streamlit App


def bot():
    pass


def get_response(query):
    pass


def main():
    st.set_page_config(page_title="Multiple PDFs chat with DeciLM-6b and LangChain",
                       page_icon=":file_folder:")

    st.header("Chaat with Multiple PDFs chat using DeciLM-6b and LangChain")

    query = st.text_input(label="Type your question based on the PDFs",
                          placeholder="Type question...")

    if query:
        st.write(f"You asked {query}")

    with st.sidebar:
        st.subheader("Hello, welcome!")

        pdfs = st.file_uploader(label="Upload your PDFs here and click Process!",
                                accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing file(s)..."):
                vector_db = creat_vector_db(pdfs)
                prompt = define_custom_prompt()

                conversation = conversation_chain(vector_db)

                st.write(conversation)

            st.write("Your files are Processed. You set to ask questions!")


if __name__ == "__main__":
    main()
