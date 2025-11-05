import streamlit as st
from information_retrieval.utils import chunk_documents, extract_pdf_to_documents
from information_retrieval.utils import get_vector_store, build_conversational_chain
from information_retrieval.log_config import setup_logging


def main():

    # ------------------------------------
    #  Logger Setup
    # ------------------------------------
    if "logger_initialized" not in st.session_state:
        logger = setup_logging()
        st.session_state.logger_initialized = True
        logger.info("Application started")
    else:
        import logging
        logger = logging.getLogger("InformationRetrievalApp")
    
    # ------------------------------------
    #  Page Config and Header
    # ------------------------------------
    st.set_page_config("Information Retrieval App")
    st.header("Information Retrieval Application")
    st.markdown("Welcome! \n\nThis Information Retreival Applications lets you interact with your documents in a conversational way."
    "Simply upload one or more PDF files, and then ask questions in natural language."
    "Behind the scenes, your documents are converted into vector embedding, enabling semantic search and conversational responses based on their content.")

    # ------------------------------------
    #  Initialize Session State
    # ------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # if  st.session_state.conversation is None and st.session_state.vector_store is not None:
    #     st.session_state.conversation = build_conversational_chain(st.session_state.vector_store)
        
    # ------------------------------------
    #  Sidebar (always available) 
    # ------------------------------------
    # Adding a side panel on the webpage and let the users upload PDF files.
    with st.sidebar:
        uploaded_file = st.file_uploader("**Upload PDF Files:**", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit"):
            with st.spinner("Processing Files..."):
                try:
                    extracted_text = extract_pdf_to_documents(uploaded_file)
                    extracted_chunks = chunk_documents(extracted_text)
                    st.session_state.vector_store = get_vector_store(extracted_chunks)
                    st.session_state.conversation = build_conversational_chain(st.session_state.vector_store)
                    st.session_state.chat_history = []
                    st.success("PDF processed Successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF files: {e}")
    
    # ------------------------------------
    #  Chat Input
    # ------------------------------------
    user_question = st.chat_input("Ask a Question from the uploaded PDF Files")

    # -----------------------------------------
    # Render Existing Chat History first
    # ------------------------------------------
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).markdown(message["content"])

    if user_question:
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        if st.session_state.conversation is None:
            st.chat_message("assistant").markdown("⚠️ Please upload and process a PDF first.")
        else:
            try:
                assistant_placeholder = st.chat_message("assistant")
                result_placeholder = assistant_placeholder.empty()
                result_placeholder.markdown("⏳ Analyzing...")
                response = st.session_state.conversation({"question": user_question})
                answer = response["answer"]
                result_placeholder.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                result_placeholder.markdown(f"Error: {e}")

    
if __name__ == "__main__":
    main()
