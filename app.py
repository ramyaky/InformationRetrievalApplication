import streamlit as st
from information_retrieval.utils import chunk_documents, extract_pdf_to_documents
from information_retrieval.utils import get_vector_store
from information_retrieval.log_config import setup_logging


def main():

    if "logger_initialized" not in st.session_state:
        logger = setup_logging()
        st.session_state.logger_initialized = True
        logger.info("Application started")
    else:
        import logging
        logger = logging.getLogger("InformationRetrievalApp")
    
    st.set_page_config("Information Retrieval App")
    st.header("Information Retrieval Application")

    extracted_text = ""

    # Adding a side panel on the webpage and let the users upload PDF files.
    with st.sidebar:
        uploaded_file = st.file_uploader("Please upload your PDF files and click Submit button: ", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit"):
            with st.spinner("Processing Files..."):
                extracted_text = extract_pdf_to_documents(uploaded_file)
                extracted_chunks = chunk_documents(extracted_text)
                vector_store = get_vector_store(extracted_chunks)
                st.success("Done")
    
    st.write(extracted_text)

if __name__ == "__main__":
    main()
