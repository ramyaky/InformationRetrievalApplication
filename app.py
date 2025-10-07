import streamlit as st
from information_retrieval.utils import extract_pdf_text, extract_text_chunks

def main():
    st.set_page_config("Information Retrieval App")
    st.header("Information Retrieval Application")

    extracted_text = ""

    # Adding a side panel on the webpage and let the users upload PDF files.
    with st.sidebar:
        uploaded_file = st.file_uploader("Please upload your PDF files and click Submit button: ", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit"):
            with st.spinner("Processing Files..."):
                extracted_text = extract_pdf_text(uploaded_file)

                st.success("Done")
    
    st.write(extracted_text)

if __name__ == "__main__":
    main()
