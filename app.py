import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MED Virtual Agent", page_icon="🚢", layout="wide")

# API KEY
# Esta línea es la "llave maestra"
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("❌ No se encontró la GOOGLE_API_KEY. Configúrala en el panel de Render.")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚓ MED Virtual Agent")
    st.markdown("---")
    st.info("""
    **📜 Usage Note:** I am a RAG-based MED Expert. My Master **Francisco Broissin** has designed me, created and trained, and thus I'm optimized for **semantic analysis** of the MED Directive.
    """)

# --- INICIALIZACIÓN DEL SISTEMA ---


# --- 1. FUNCIÓN DE APOYO (FUERA) ---
def get_vector_db(embeddings):
    CHROMA_PATH = "./chroma_db"
    DOCS_PATH = "./documentos"
    
    #intento de carga	
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            db.get(limit=1) 
            print("✅ Base de datos física cargada.")
            return db
        except Exception as e:
            print(f"⚠️ Fallo en carga física: {e}. Reconstruyendo...")
    
    # Reconstrucción si lo anterior falla
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        raise Exception("No hay PDFs en la carpeta 'documentos' para reconstruir.")

    loaders = [PyPDFLoader(os.path.join(DOCS_PATH, f)) for f in os.listdir(DOCS_PATH) if f.endswith('.pdf')]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print("✅ Base de datos creada desde PDFs.")
    return db

# --- 2. FUNCIÓN PRINCIPAL ---
@st.cache_resource
def inicializar_sistema():
    try:
        # A. Configuración inicial
        api_key = st.secrets["GOOGLE_API_KEY"]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # B. Obtener la Base de Datos (usando nuestra función de apoyo)
        vectorstore = get_vector_db(embeddings)
        
        # C. Configurar el Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # D. Configurar el Modelo LLM (Gemini)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)

        # E. Prompt Template
        template = """I am the Pancho's MED Virtual Agent, a technical expert in the Marine Equipment Directive (MED).
        Answer the question based on the provided context.
        Context: {context}
        Question: {question}
        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # F. Crear la cadena QA (El motor final)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        return qa_chain, "✅ System ready"

    except Exception as e:
        return None, f"❌ Error crítico: {str(e)}"

# --- CUERPO PRINCIPAL ---
st.title("🚢 Pancho's MED Virtual Agent")

# Arrancamos el motor
qa_chain, mensaje = inicializar_sistema()

if qa_chain is None:
    st.error(mensaje)
    st.stop()
else:
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.success(mensaje)

# --- HISTORIAL Y CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Can I help you on MED Topic? Please kindly set your question here?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                respuesta_texto = response["result"]
                st.markdown(respuesta_texto)
                
                with st.expander("Ver fuentes consultadas"):
                    for doc in response["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error(f"Error: {e}")
