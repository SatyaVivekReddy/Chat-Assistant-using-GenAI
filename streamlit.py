import streamlit as st
from utils import process_csv, get_vector_store, user_input

st.set_page_config(page_title="Chat Assistant", layout="wide")

def main():
    st.header("ChatAssistant💁")
    user_question = st.text_input("Ask Product Realted Queries", key="user_question")
    if user_question:  # Ensure API key and user question are provided
        st.write(user_input(user_question))

    with st.sidebar:
        st.title("Menu:")
        csv_docs = st.file_uploader("Upload the CSV Files and Click on the Submit & Process Button",  type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                text_chunks = process_csv(csv_docs)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

