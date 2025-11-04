from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.schema import Document
from pathlib import Path
import logging
import os


logger = logging.getLogger("InformationRetrievalApp")

# Load the environment variables
load_dotenv()

#
# Iterates each PDF file and extract text from it. 
# Creates a Document object using both extracted text and some metadata. 
# Return a list of Document objects.
#
def extract_pdf_to_documents(pdf_documents):
    all_documents = []
    for pdf in pdf_documents:
        logger.debug(f"Processing PDF: {pdf.name}")
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            logger.debug(f"Extracting text from Page {page_num}")
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
    logger.debug(f"Performing text chunking with chunk size {splitter._chunk_size} and chunk overlap {splitter._chunk_overlap}")
    text_chunks = splitter.split_text(extracted_text)
    return text_chunks

#
# Splits the documents into chunks of size 1000 characters with a overlap of 200 characters.
#
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logger.debug(f"Performing Document chunking with chunk size {splitter._chunk_size} and chunk overlap {splitter._chunk_overlap}")
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
        logger.info(f"Loading Gemma Ebedding from Local Cache...")
        print("Loading Gemma Ebedding from Local Cache...")
        model_path = str(LOCAL_MODEL_PATH)
    else:
        if not HF_TOKEN:
            raise ValueError("HuggingFace Token required to download gated model..")
        
        logger.info(f"Local model not found, downloading from HuggingFace  Hub..")       
        print("Local model not found, downloading from HuggingFace  Hub...")
        model_path = "google/embeddinggemma-300m"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"use_auth_token": HF_TOKEN}  #token ignored if local model exists.
    )
    logger.info(f"Successfully loaded embedding model")
    return embeddings


#
# Fetches embedding model and creates vector store.
# Uses embeddings to convert extracted text chunks into dense vectors.
#
def get_vector_store(extracted_chunks, persist_dir="vector_stores/indexes"):
    embeddings = get_embedding_model()
    
    # Check if vectors are already processed, just load from the disk.
    if os.path.exists(persist_dir):
        logger.info("Local vector store already exists. Loading..")
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    
    logger.info("Could not find any saved local vector stores. Creating one..")
    # Creates FAISS vector store from text chunks
    logger.debug(f"Converting chunks to embedding vectors")
    vector_store = FAISS.from_documents(extracted_chunks, embeddings)
    
    logger.debug(f"Saving vector store index locally under vector_stores/indexes folder")
    vector_store.save_local(persist_dir)
    
    logger.debug(f"Successfully saved the vector store index.")
    return vector_store


def build_conversational_chain(vector_index):
    logger.info("Connecting to Ollama at model=llama3 on http://localhost:11434")

    # Create ChatOllama LLM wrapper
    # llm = ChatOllama(
    #         model="llama3", 
    #         base_url="http://localhost:11434",
    #         temperature=0.0,  # Dont be creative. Just give me the most likely and accurate answer 
    #         disable_streaming=True
    #     )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.2
    )
    
    # Convert vector store as retriever to perfrom search.
    retriever = vector_index.as_retriever(search_kwargs={"k": 4})

    # Conversation memory (keeps turn history)
    memory = ConversationBufferMemory(memory_key="chat_history", 
                                      return_messages=True)

    # Build chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                  retriever=retriever, 
                                                  memory=memory)

    return chain
