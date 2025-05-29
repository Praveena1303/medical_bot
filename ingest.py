import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def update_vector_db_with_new_pdf():
    # Verify PDF exists
    pdf_path = os.path.join(DATA_PATH, "A-Z Family Medical Encyclopedia.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {os.path.abspath(pdf_path)}")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    # Load existing DB or create new
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        from langchain_community.vectorstores.utils import DistanceStrategy
        db = FAISS.from_texts(
            ["Initial document"], 
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )

    # Process PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    db.add_documents(texts)
    db.save_local(DB_FAISS_PATH)
    print(f"Successfully updated index with {len(texts)} chunks")

if __name__ == "__main__":
    print("Running ingestion...")
    update_vector_db_with_new_pdf()