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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MED Virtual Agent", page_icon="🚢", layout="wide")

# API KEY
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ No se encontró la GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=api_key)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚓ MED Virtual Agent")
    st.markdown("---")
    st.info("""
    **📜 Usage Note:** I am a RAG-based MED Expert. Optimized for **semantic analysis** of the MED Directive.
    """)

# --- INICIALIZACIÓN DEL SISTEMA ---
@st.cache_resource
def inicializar_sistema():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(base_path, "chroma_db")
        
        # 1. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Verificación de Carpeta
        if not os.path.exists(persist_dir):
            return None, f"❌ Carpeta '{persist_dir}' no encontrada."

        # 3. Carga de DB (con espacios consistentes)
        vectorstore = Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings,
            collection_name="langchain"
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. Modelo LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)
        
        # 5. Prompt Template
        template = """I am the Pancho's MED Virtual Agent, a technical expert in the Marine Equipment Directive (MED).
        Answer the question based on the provided context.
        Context: {context}
        Question: {question}
        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # 6. Crear la cadena QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return qa_chain, "✅ Sistema listo"
    
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

if prompt := st.chat_input("¿En qué puedo ayudarte?"):
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
