import streamlit as st
from pypdf import PdfReader
import torch
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# load the environments
from dotenv import load_dotenv
load_dotenv()



#DEFINE SOME VARIABLES

CHUNK_SIZE = 1000
# Using HuggingFaceEmbeddings with the chosen embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs = {"device": "cuda"})



# transformer model configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)



# CREATE A VECTOR DATABASE - FAISS
def creat_vector_db(uploaded_pdfs) -> FAISS:
    """Read multiple PDFs, split, embedd and store the embeddings on FAISS vector store"""

    text = ""  
    for pdf in uploaded_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                   chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    vector_db = FAISS.from_texts(texts, embeddings) # create vector db for similarity search    
    vector_db.save_local("faiss_index") # save the vector db to avoid repeated calls to it
    return vector_db

# LOAD LLM
def load_llm():

    model_id = "Deci/DeciLM-6b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            trust_remote_code=True, 
                                            device_map = "auto",
                                            quantization_config=quant_config)

    pipe = pipeline("text-generation", 
                    model=model,
                    tokenizer=tokenizer,
                    temperature=0.1,
                    return_full_text = True, 
                    max_new_tokens=40,
                    repetition_penalty =  1.1)
    
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm



# RESPONSE INSTRUCTIONS
def set_custom_prompt():
    """instructions, to the llm for text response generation"""


    custom_prompt_template = """You have been given the following documents to answer the user's question.
    If you do not have information from the information given to answer the questions just say 'I don't know the answer" and don't try to make up an answer.
    Context: {context}
    Question: {question}
    Give a detailed helpful answer and nothing more.
    Helpful answer:
"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
                            "context", "question"])
    return prompt



# QUESTION ANSWERING CHAIN
def retrieval_qa_chain(prompt, vector_db):
    """Chain to retrieve answers. the chain takes  the documents and
    makes a call to the DeciLM-6b llm """

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

# QUESTION ANSWER BOT
def qa_bot():  
    vectore_db = FAISS.load_local("faiss_index", embeddings)  
    conversation_prompt = set_custom_prompt()
    conversation = retrieval_qa_chain(conversation_prompt, vectore_db)
    return conversation

# RESPONSE FROM BOT
def bot_response(query):
    conversation_result = qa_bot()
    response = conversation_result({"query": query})
    return response["result"]


def main():
    st.set_page_config(page_title="Multiple PDFs chat with DeciLM-6b and LangChain",
                       page_icon=":file_folder:")

    # page side panel
    with st.sidebar:
        st.subheader("Hello, welcome!")

        pdfs = st.file_uploader(label="Upload your PDFs here and click Process!",
                                accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing file(s)..."):
                # create a vectore store
                creat_vector_db(pdfs)
            st.write("Your files are Processed. You set to ask questions!")

    st.header("Chat with Multiple PDFs using DeciLM-6b-instruct LLM")

    # Query side
    query = st.text_input(label="Type your question based on the PDFs",
                          placeholder="Type question...")

    if query:
        st.write(f"Query: {query}")
        st.text(textwrap.fill(bot_response(query), width=80))


if __name__ == "__main__":
    main()