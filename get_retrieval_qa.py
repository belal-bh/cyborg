from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.vectorstores import Chroma
from load_model import load_model

from config import (
    EMBEDDING_MODEL_NAME,
    KB_DIR,
    CHROMA_SETTINGS,
    LLM_MODEL_ID,
    LLM_MODEL_BASENAME
)


def get_retrieval_qa(device_type):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # uncomment the following line if you used HuggingFaceEmbeddings in the generateKB.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=KB_DIR,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    llm = load_model(device_type, model_id=LLM_MODEL_ID,
                     model_basename=LLM_MODEL_BASENAME)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    return qa
