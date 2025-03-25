import os

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

template = """
تو یک دستیار هستی که از یک داده های متنی و تصویری استفاده میکنی تا به سوالات کاربر به زبان فارسی سلیس پاسخ بدی.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'multi-modal-rag/pdfs/'
figures_directory = 'multi-modal-rag/figures/'
images_directory = 'multi-modal-rag/images/'

# Create directories if they don't exist
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs(figures_directory, exist_ok=True)
os.makedirs(images_directory, exist_ok=True)

embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="gemma3")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def upload_image(file):
    with open(images_directory + file.name, "wb") as f:
        f.write(file.getbuffer())
    return images_directory + file.name

def load_pdf(file_path):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]

    for file in os.listdir(figures_directory):
        extracted_text = extract_text(figures_directory + file)
        text_elements.append(extracted_text)

    return "\n\n".join(text_elements)

def extract_text(file_path):
    model_with_image_context = model.bind(images=[file_path])
    return model_with_image_context.invoke("Tell me what do you see in this picture.")

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_text(text)

def index_docs(texts):
    vector_store.add_texts(texts)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

# Sidebar for upload options
st.sidebar.title("Upload Documents")
upload_option = st.sidebar.radio("Choose upload type:", ["PDF", "Image"])

if upload_option == "PDF":
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=False
    )

    if uploaded_file:
        upload_pdf(uploaded_file)
        text = load_pdf(pdfs_directory + uploaded_file.name)
        chunked_texts = split_text(text)
        index_docs(chunked_texts)
else:
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_image:
        image_path = upload_image(uploaded_image)
        st.image(image_path, caption="Uploaded Image", use_column_width=True)
        image_description = extract_text(image_path)
        index_docs([image_description])
        st.write("Image processed and added to knowledge base")

# Chat interface
question = st.chat_input()

if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)
    st.chat_message("assistant").write(answer)