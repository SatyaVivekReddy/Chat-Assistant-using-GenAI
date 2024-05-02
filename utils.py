import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDc08P8wVaPI01W-3eMtmQErjQSIasnLhw"

# Instantiate GoogleGenerativeAIEmbeddings once
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="retrieval_document",
)

def read_csv(file_path):
    return pd.read_csv(file_path)

def get_chunks_from_csv(dataframe):
    chunks = []
    for index, row in dataframe.iterrows():
        chunk = f"{row['product_name']} {row['category']} {row['discounted_price']} {row['actual_price']} {row['rating']} {row['about_product']} {row['review_content']}"
        chunks.append(chunk)
    return chunks

def process_csv(file_path):
    dataframe = read_csv(file_path)
    chunks = get_chunks_from_csv(dataframe)
    return chunks

def get_vector_store(chunks):
    # Use the previously instantiated embeddings
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]




