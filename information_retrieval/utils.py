from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from pathlib import Path
import os


#
# Iterates each PDF file and extract text from it. 
# Creates a Document object using both extracted text and some metadata. 
# Return a list of Document objects.
#
def extract_pdf_to_documents(pdf_documents):
    all_documents = []
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()  # extracts text out of pdf. You can just concat this string to use text chunking instead of document chunking.
            if text:
                all_documents.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": pdf.name}
                ))
    return all_documents

#
# Splits the text into chunks of size 1000 characters with a overlap of 20 characters.
#
def chunk_text(extracted_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    text_chunks = splitter.split_text(extracted_text)
    return text_chunks

#
# Splits the documents into chunks of size 1000 characters with a overlap of 200 characters.
#
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = splitter.split_documents(documents)

    return document_chunks

#
# Function to load local/already downloaded embeddinggemma-300m model. 
# If not downloads from huggingface hub.
#
def get_embedding_model():
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    LOCAL_MODEL_PATH = Path("./models/google/embeddinggemma-300m")

    if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
        print("Loading Gemma Ebedding from Local Cache...")
        model_path = str(LOCAL_MODEL_PATH)
    else:
        if not HF_TOKEN:
            raise ValueError("HuggingFace Token required to download gated model..")
        print("Local model not found, downloading from HuggingFace  Hub...")
        model_path = "google/embeddinggemma-300m"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"use_auth_token": HF_TOKEN}  #token ignored if local model exists.
    )

    return embeddings


#
# Fetches embedding model and creates vector store.
# Uses embeddings to convert extracted text chunks into dense vectors.
#
def get_vector_store(extracted_chunks):
    embeddings = get_embedding_model()
    # Creates FAISS vector store from text chunks
    vector_store = FAISS.from_documents(extracted_chunks, embeddings)
    vector_store.save_local("indexes/sample_index")
    return vector_store
