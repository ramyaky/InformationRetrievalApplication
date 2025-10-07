from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
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
def extract_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    chunks = splitter.split_text(text)
    return chunks
