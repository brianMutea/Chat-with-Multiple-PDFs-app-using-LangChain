import streamlit as st
import pypdf
import time
import tiktoken
from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, DeepSparse
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

from transformers import AutoModelForCausalLM, AutoTokenizer


from htmlTemplates import css, user_template, bot_template


from dotenv import load_dotenv
load_dotenv()


def get_pdf_text(uploaded_pdfs):

    text = ""

    # read the file first

    for pdf in uploaded_pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_texts(
        texts=text_chunks, embedding=embeddings)  # db

    return vector_store


def load_llm():
    model_id = 'Deci/DeciLM-6b'

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True
                                                 )
    llm = DeepSparse(
        model=model,
        model_config={"sequence_length": 1000, "trust_remote_code": True},
        generation_config={"max_new_tokens": 300}
    )

    return llm


def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    # llm = OpenAI(temperature=0.7)

    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory)

    # return conversation_chain

    llm = load_llm()

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=vectorstore.as_retriever())
    return qa_chain


def handle_userQuery(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


def main():

    st.set_page_config(page_title="Chat with Multiple PDFs",
                       page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with Multiple PDFs :books:")

    user_question = st.text_input(
        label="Ask a question about your PDFs", placeholder="Type question...")

    if user_question:
        st.write(user_question)
        handle_userQuery(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hellow robot?"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "Hellow human?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click Process!", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing your files..."):
                # get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
            st.write("Your files have been processed!")


if __name__ == '__main__':
    main()
