from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chainlit as cl
from auth import load_users
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = 'vectorstore/db_faiss'
custom_prompt_template = """Use the medical context to answer professionally.
If unsure, say you don't know. Don't invent answers.

Context: {context}
Question: {question}

Professional medical answer:"""

# Password authentication callback
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    users = load_users()
    if username in users and users[username] == password:
        return cl.User(identifier=username)
    return None

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

def load_llm():
    # Verify model path exists
    model_path = "C:/Users/Gaming/Desktop/models/llama-2-7b-chat.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return CTransformers(
        model=model_path,
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.5}
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Verify FAISS index exists
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found at {DB_FAISS_PATH}")
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        chain_type_kwargs={'prompt': set_custom_prompt()},
        return_source_documents=True
    )
    return qa

@cl.on_chat_start
async def start():
    try:
        chain = qa_bot()
        msg = cl.Message(content="Authenticating...")
        await msg.send()
        msg.content = "Verified. How can I assist with medical queries today?"
        await msg.update()
        cl.user_session.set("chain", chain)
    except Exception as e:
        await cl.Message(content=f"Error initializing bot: {str(e)}").send()
        raise

@cl.on_message
async def main(message: cl.Message):
    try:
        chain = cl.user_session.get("chain")
        cb = cl.AsyncLangchainCallbackHandler()
        res = await chain.ainvoke(
            {"query": message.content},
            callbacks=[cb]
        )
        answer = res["result"]
        sources = res["source_documents"]
        
        if sources:
            answer += f"\n\nSources: {sources[0].metadata['source']}"
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"Error processing your query: {str(e)}").send()