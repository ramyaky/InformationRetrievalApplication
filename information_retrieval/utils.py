from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os


#
# Iterates each PDF file and extract text from it. Returns the consolidated extracted text.
#
def extract_pdf_text(pdf_documents):
    extracted_text = ""
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
    
    return extracted_text

#
# Splits the text into chunks of size 1000 characters with a overlap of 20 characters.
#
def extract_text_chunks(extracted_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    chunks = splitter.split_text(extracted_text)
    return chunks

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