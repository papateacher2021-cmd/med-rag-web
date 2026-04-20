import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="MED Virtual Agent", page_icon="🚢", layout="wide")

# --- 2. ENCABEZADO Y TEXTOS DE PANCHO ---
col1, col2 = st.columns([7, 3])

with col1:
    st.title("🚢 Pancho's MED Virtual Agent (V2.5)")
    st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para consultar la **Directiva de Equipos Marinos (MED)** y las **Recomendaciones MarED**.
""")
    st.info("""
**Master Pancho's Note:** I have instructed this agent with only documents and references I trust. 
No worries if it does not know something; it will say so.
""")
    st.write("""
I am a RAG Agent for MED. I have learned: **MED**, **MarED GEN-001** and **MarED GEN-010**.
""")
    st.caption("© 2026 Francisco Brossin. MIT License. Source code will be on GitHub in due time.")

with col2:
    image_path = 'rag-diagram.png'
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Arquitectura RAG', use_container_width=True)

st.divider()

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
    **📜 Usage Note:** I am a RAG-based MED Expert Virtual Agent. My creator. my Master Francisco Broissin has trained me  and optimized for **semantic analysis** of the MED Directive and MarED Recommendations.

	I've been designed to assist in research and cross-referencing concepts. I'm **not** intended for literal transcription of legal articles. For exact legal quotes or compliance citations, please always refer to the official source documents.

    """)

# --- INICIALIZACIÓN DEL SISTEMA (FUNCIÓN ÚNICA) ---
@st.cache_resource
def inicializar_sistema():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(base_path, "chroma_db")
        
        # 1. Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Verificación de Carpeta
        if not os.path.exists(persist_dir):
            return None, f"❌ Carpeta '{persist_dir}' no encontrada. Archivos: {os.listdir(base_path)}"

        # 3. Carga de DB
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
        
        # 4. Modelo LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)
        
        # 5. Prompt Template
        template = """I am the Pancho's MED Virtual Agent, a technical expert in the Marine Equipment Directive (MED).
        Answer the question based on the provided context.
        Context: {context}
        Question: {question}
        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # 6. Crear la cadena QA (Aquí es donde vivía el error antes)
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
st.title("🚢 Pancho's MED Virtual Agent (V2.5)")

# Arrancamos el motor
qa_chain, mensaje = inicializar_sistema()

if qa_chain is None:
    st.error(mensaje)
    st.stop()
else:
    # Mostramos éxito solo si el historial está vacío para no ensuciar
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        st.success(mensaje)

# --- HISTORIAL Y CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En qué puedo ayudarte con la normativa MED?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manuales..."):
            try:
                # Usamos el qa_chain que inicializamos arriba
                response = qa_chain.invoke({"query": prompt})
                respuesta_texto = response["result"]
                st.markdown(respuesta_texto)
                
                with st.expander("Ver fuentes consultadas"):
                    for doc in response["source_documents"]:
                        st.caption(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            except Exception as e:
                st.error(f"Error: {e}")
