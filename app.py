import streamlit as st
import os
import glob
import shutil
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from dotenv import load_dotenv
import time
import json

# ==========================================
# 🛡️ IMPORTACIONES ULTRA-ROBUSTAS (Classic-Aware)
# ==========================================

try:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
    except ImportError:
        try:
            from langchain_community.memory import ConversationBufferMemory
            from langchain_community.chains import ConversationalRetrievalChain
        except ImportError as e:
            st.error(f"❌ Error Crítico: No se pudo cargar los módulos de memoria de LangChain.")
            st.stop()

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.vectorstores import Chroma
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        from langchain.text_splitter import CharacterTextSplitter
    except:
        st.error("Error al cargar extensiones de LangChain.")
        st.stop()

# --- ICONOGRAFÍA PREMIUM (SVG) ---
ICON_BRAIN = """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54Z"/></svg>"""
ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>"""
ICON_CHART = """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>"""
ICON_LAYERS = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>"""
ICON_DATA = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>"""

# --- CONFIGURACIÓN UI ---
st.set_page_config(page_title="Brain Balance RAG", page_icon="🧠", layout="wide")

PALETTE = {
    "operaciones": "#6366f1",
    "seguridad": "#f43f5e",
    "plataformas": "#10b981",
    "empresa": "#8b5cf6",
    "hit": "#22c55e",  # Verde para éxito
    "miss": "#ef4444"   # Rojo para fallo
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    /* Core Layout */
    html, body, [class*="st-"] {{ 
        font-family: 'Plus Jakarta Sans', sans-serif; 
        background-color: #fcfdfe; 
        color: #1e293b; 
    }}
    
    /* Header */
    .dashboard-header {{ 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); 
        padding: 2.5rem 3rem; 
        border-radius: 0 0 40px 40px; 
        border-bottom: 1px solid rgba(226, 232, 240, 0.8); 
        margin-bottom: 2.5rem; 
        display: flex; 
        align-items: center; 
        gap: 2rem; 
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.05); 
    }}
    .icon-container-main {{ 
        background: linear-gradient(135deg, #0f172a 0%, #334155 100%); 
        color: #ffffff; 
        padding: 1rem; 
        border-radius: 20px; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.2);
    }}
    .dashboard-title {{ 
        font-size: 2.5rem; 
        font-weight: 800; 
        color: #0f172a; 
        margin: 0; 
        letter-spacing: -1.5px; 
        line-height: 1.1; 
    }}
    
    /* Cards */
    .info-card {{ 
        background: #ffffff; 
        border: 1px solid rgba(226, 232, 240, 0.6); 
        border-radius: 30px; 
        padding: 2rem; 
        height: 100%; 
        margin-bottom: 1.5rem; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.04);
        border-color: rgba(99, 102, 241, 0.3);
    }}
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{ 
        background-color: #f1f5f9; 
        padding: 10px; 
        border-radius: 24px; 
        gap: 12px; 
    }}
    .stTabs [data-baseweb="tab"] {{ 
        color: #64748b; 
        border-radius: 16px; 
        padding: 12px 30px; 
        font-weight: 700; 
        font-size: 0.9rem;
        transition: all 0.2s ease;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: #ffffff !important; 
        color: #0f172a !important; 
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); 
    }}
    
    /* Premium Widgets */
    .tech-pill {{ 
        background: #f8fafc; 
        color: #475569; 
        padding: 8px 16px; 
        border-radius: 100px; 
        font-size: 0.75rem; 
        font-weight: 700; 
        border: 1px solid #e2e8f0; 
        margin: 6px; 
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    
    /* Explanation Box */
    .explanation-box {{ 
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%); 
        border: 1px solid rgba(226, 232, 240, 0.8); 
        border-radius: 35px; 
        padding: 2.5rem; 
        margin-top: 2rem; 
        box-shadow: 0 20px 50px rgba(0,0,0,0.03); 
        position: relative;
        overflow: hidden;
    }}
    .explanation-box::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 8px;
        height: 100%;
        background: linear-gradient(to bottom, #6366f1, #a855f7);
    }}
    .explanation-box h4 {{
        color: #0f172a;
        margin-top: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}
    
    /* Code Snippets */
    .code-snippet {{ 
        background: #0f172a; 
        color: #e2e8f0; 
        padding: 24px; 
        border-radius: 20px; 
        font-family: 'JetBrains Mono', 'Fira Code', monospace; 
        font-size: 0.8rem; 
        margin: 1.5rem 0; 
        overflow-x: auto; 
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
</style>
""", unsafe_allow_html=True)

# --- PERSISTENCIA ---
LOG_FILE = "brain_persistence_log.json"
def save_logs():
    data = {"chat_history": st.session_state.chat_history, "query_trace_data": st.session_state.query_trace_data, "topic_counts": st.session_state.topic_counts}
    with open(LOG_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)
def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return None
    return None

# --- INICIALIZACIÓN ---
persisted = load_logs()
if "chat_history" not in st.session_state: st.session_state.chat_history = persisted["chat_history"] if persisted else []
if "topic_counts" not in st.session_state: st.session_state.topic_counts = persisted["topic_counts"] if persisted else {"operaciones": 0, "seguridad": 0, "plataformas": 0, "empresa": 0}
if "query_trace_data" not in st.session_state: st.session_state.query_trace_data = persisted["query_trace_data"] if persisted else []

load_dotenv()
KNOWLEDGE_BASE = "knowledge-base"

def get_topic_from_answer(result):
    ans = result.get("answer", "").lower()
    source_docs = result.get("source_documents", [])
    
    # Lista mejorada de frases de negación o desconocimiento
    negative_phrases = [
        "no tengo información", "no se menciona", "no puedo responder", 
        "no hay datos", "lo siento", "no se encuentra", "no sé", "no se", 
        "desconozco", "no consta", "no hay registro", "no hay mención",
        "no tengo detalles", "lo lamento", "no dispongo", "no se especifica"
    ]
    
    # Si la respuesta empieza con negación o es demasiado corta, es un MISS asegurado
    is_neg = False
    clean_ans = ans.strip()
    if any(clean_ans.startswith(p) for p in negative_phrases) or len(clean_ans) < 30:
        is_neg = True

    if not source_docs or is_neg: 
        return None, False
        
    counts = {"operaciones": 0, "seguridad": 0, "plataformas": 0, "empresa": 0}
    for doc in source_docs:
        dtype = doc.metadata.get("doc_type", "unknown")
        if dtype in counts: counts[dtype] += 1
    
    # Si ninguna de las fuentes coincide con las categorías de Brain Balance, es un miss
    if sum(counts.values()) == 0: return None, False
    
    return max(counts, key=counts.get), True

@st.cache_resource
def init_rag_system():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("⚠️ OPENAI_API_KEY no detectada. Por favor, configúrela como una variable de entorno.")
            return None, None, [], []
            
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        all_docs = []
        for f in glob.glob(f"{KNOWLEDGE_BASE}/*"):
            dtype = os.path.basename(f)
            loader = DirectoryLoader(f, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
            docs = loader.load()
            for d in docs: d.metadata["doc_type"] = dtype
            all_docs.extend(docs)
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=120) 
        chunks = splitter.split_documents(all_docs)
        vstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=os.path.abspath("vector_db_brain_balance"))
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        retriever = vstore.as_retriever(search_kwargs={"k": 5})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer') 
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True)
        return chain, vstore, all_docs, chunks
    except Exception as e:
        st.error(f"Error Initialize: {e}"); return None, None, [], []

if "chain" not in st.session_state:
    with st.spinner("Inicializando Brain Balance Engine..."):
        st.session_state.chain, st.session_state.vectorstore, st.session_state.docs, st.session_state.chunks = init_rag_system()

# --- HEADER ---
st.markdown(f"""
<div class="dashboard-header">
    <div class="icon-container-main">{ICON_BRAIN}</div>
    <div><div class="dashboard-title">Brain Balance Showcase</div><div style="color: #64748b; font-weight: 600; font-size: 1rem; margin-top: 5px;">RAG Intelligence Platform • v2.1 Performance</div></div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_kb, tab_queries, tab_docs, tab_stack = st.tabs(["💬 Asistente RAG", "🗺️ Mapa Conocimiento", "🎯 Trazado Queries", "📊 Gestión Datos", "⚙️ Stack Técnico"])

# --- TAB 1: CHAT ---
with tab_chat:
    c1, c2 = st.columns([2.2, 1])
    with c1:
        container = st.container(height=520)
        if not st.session_state.chat_history:
            with container.chat_message("assistant", avatar="🧠"):
                st.markdown("### 👋 Hola, soy el Experto en Conocimiento de Brain Balance")
                st.markdown("""
                Utilizo tecnología **RAG (Retrieval Augmented Generation)** para responderte de forma verídica.
                
                **¿Qué es RAG?** Es un sistema que, antes de responder, busca en nuestros manuales internos para encontrar el contexto exacto. 
                Esto evita que la IA alucine o invente datos, garantizando que cada respuesta esté anclada en la realidad de la empresa.
                """)
        for m in st.session_state.chat_history:
            av = "🧠" if m["role"] == "assistant" else "👤"
            with container.chat_message(m["role"], avatar=av): st.markdown(m["content"])
        
        if q := st.chat_input("Consulta a la base de conocimiento..."):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with container.chat_message("user", avatar="👤"): st.markdown(q)
            with container.chat_message("assistant", avatar="🧠"):
                if st.session_state.chain:
                    res = st.session_state.chain.invoke({"question": q})
                    ans = res["answer"]; st.markdown(ans)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    topic, success = get_topic_from_answer(res)
                    try:
                        api_key = os.getenv("OPENAI_API_KEY")
                        q_emb = OpenAIEmbeddings(openai_api_key=api_key).embed_query(q)
                        st.session_state.query_trace_data.append({"text": q, "embedding": q_emb, "topic": topic if success else "miss", "success": success})
                    except: pass
                    if success: st.session_state.topic_counts[topic] += 1
                    save_logs(); st.rerun()

    with c2:
        st.markdown(f"<div class='info-card'><h4>{ICON_TARGET} Consultas Sugeridas</h4>", unsafe_allow_html=True)
        # Sugerencias basadas en los documentos detectados (empresa, seguridad, plataformas, operaciones)
        suggestions = [
            "¿Cómo nació Brain Balance y cuál es su misión?",
            "¿Cuáles son las comisiones para retiros por PayPal y Payoneer?",
            "¿Qué medidas de seguridad debo seguir para evitar el Phishing?",
            "¿Cómo funciona el intercambio de criptomonedas en la plataforma?"
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": s})
                if st.session_state.chain:
                    with st.spinner("Consultando..."):
                        res = st.session_state.chain.invoke({"question": s})
                        ans = res["answer"]
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                        topic, success = get_topic_from_answer(res)
                        try:
                            api_key = os.getenv("OPENAI_API_KEY")
                            q_emb = OpenAIEmbeddings(openai_api_key=api_key).embed_query(s)
                            st.session_state.query_trace_data.append({"text": s, "embedding": q_emb, "topic": topic if success else "miss", "success": success})
                        except: pass
                        if success: st.session_state.topic_counts[topic] += 1
                        save_logs(); st.rerun()
        
        st.markdown("<hr style='margin: 25px 0; border: 0.5px solid #e2e8f0;'>", unsafe_allow_html=True)
        if st.button("♻️ Reiniciar Sesión", use_container_width=True):
            st.session_state.chat_history = []; st.session_state.query_trace_data = []; st.session_state.topic_counts = {k:0 for k in st.session_state.topic_counts}
            if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
            st.rerun()

# --- PROYECCIÓN GEOMÉTRICA COMPARTIDA ---
# Calculamos las coordenadas una sola vez para que ambos gráficos sean gemelos
v3kb, v3qs = None, None
if st.session_state.vectorstore:
    res_db = st.session_state.vectorstore._collection.get(include=['embeddings', 'metadatas', 'documents'])
    kb_e = np.array(res_db["embeddings"])
    qs_e = np.array([q["embedding"] for q in st.session_state.query_trace_data]) if st.session_state.query_trace_data else np.empty((0, 1536))
    
    # Unificamos Docs + Queries en el mismo cálculo t-SNE
    all_e = np.vstack([kb_e, qs_e]) if qs_e.size > 0 else kb_e
    v3_all = TSNE(n_components=3, perplexity=min(30, len(all_e)-1) if len(all_e) > 1 else 1, random_state=42, init='random').fit_transform(all_e)
    v3kb = v3_all[:len(kb_e)]
    v3qs = v3_all[len(kb_e):]

# --- TAB 2: KB MAP ---
with tab_kb:
    st.markdown(f"### {ICON_DATA} Mapa Geométrico de Conocimiento (ChromaDB)", unsafe_allow_html=True)
    if v3kb is not None:
        typs = [m.get("doc_type", "unknown") for m in res_db["metadatas"]]
        fig_kb = go.Figure()
        for t in set(typs):
            mask = [typ == t for typ in typs]
            fig_kb.add_trace(go.Scatter3d(
                x=v3kb[mask, 0], y=v3kb[mask, 1], z=v3kb[mask, 2], 
                mode='markers', 
                name=t.upper(), 
                marker=dict(size=7, color=PALETTE.get(t, "#94a3b8"), opacity=0.85, line=dict(color='white', width=0.5))
            ))
        
        fig_kb.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=0, r=0, b=0, t=0), 
            height=650,
            scene=dict(
                xaxis=dict(showticklabels=True, title="Dimensión X"),
                yaxis=dict(showticklabels=True, title="Dimensión Y"),
                zaxis=dict(showticklabels=True, title="Dimensión Z")
            )
        )
        st.plotly_chart(fig_kb, use_container_width=True)
        
        st.markdown(f"""
        <div class="explanation-box">
            <h4>📐 Cartografía del Conocimiento (Espacio Vectorial)</h4>
            <p>Este mapa es una representación tridimensional de cómo la inteligencia artificial "entiende" los manuales de <b>Brain Balance</b>. No es un gráfico estadístico convencional, sino un <b>universo semántico</b>.</p>
            <hr style="opacity: 0.1; margin: 10px 0;">
            <p><b>¿Cómo funciona este Mapa?</b></p>
            <ul>
                <li><b>Embeddings (Vectorización):</b> Cada fragmento de texto se convierte en una serie de 1,536 números que representan su significado. Estos números actúan como "coordenadas GPS" en este espacio.</li>
                <li><b>Proximidad = Relación:</b> Si dos esferas están cerca, significa que su contenido es similar. El sistema agrupa automáticamente temas de <i>Seguridad</i>, <i>Operaciones</i> o <i>Empresa</i> en nubes de puntos denominadas "clústeres".</li>
                <li><b>Navegación del LLM:</b> Cuando haces una pregunta, la IA viaja a la zona del mapa más cercana a tu consulta para leer sólo los documentos que están allí, evitando confundirse con información irrelevante de otras áreas.</li>
            </ul>
            <p>Esta tecnología permite que <b>Brain Balance</b> tenga una memoria estructurada, permitiendo auditorías rápidas sobre qué áreas de conocimiento están más pobladas y cuáles necesitan más documentación.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: TRAJECTORY SHOWCASE ---
with tab_queries:
    st.markdown(f"### {ICON_TARGET} Trayectoria Semántica de Consultas en Tiempo Real", unsafe_allow_html=True)
    if len(st.session_state.query_trace_data) > 0 and v3qs is not None:
        fig_q = go.Figure()
        
        # 1. Capa de Referencia (Docs fantasma) para mantener escala y coherencia
        typs = [m.get("doc_type", "unknown") for m in res_db["metadatas"]]
        for t in set(typs):
            mask = [typ == t for typ in typs]
            fig_q.add_trace(go.Scatter3d(
                x=v3kb[mask, 0], y=v3kb[mask, 1], z=v3kb[mask, 2], 
                mode='markers', 
                name=f"REF {t.upper()}", 
                opacity=0.02, # Casi invisible, solo para escala
                showlegend=False,
                hoverinfo='none',
                marker=dict(size=3, color=PALETTE.get(t, "#94a3b8"))
            ))

        # 2. Consultas Reales (Protagonistas con alta visibilidad)
        for i, q in enumerate(st.session_state.query_trace_data):
            color = PALETTE["hit"] if q.get("success") else PALETTE["miss"]
            status = "HIT" if q.get("success") else "MISS"
            fig_q.add_trace(go.Scatter3d(
                x=[v3qs[i, 0]], y=[v3qs[i, 1]], z=[v3qs[i, 2]], 
                mode='markers', 
                name=f"Q{i+1}: {status}", 
                text=[f"<b>CONSULTA {i+1}</b><br>{q['text']}"],
                hoverinfo="text",
                marker=dict(size=16, color=color, opacity=1.0, line=dict(color='white', width=3))
            ))

        fig_q.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=0, r=0, b=0, t=0), 
            height=650, 
            scene=dict(
                xaxis=dict(showticklabels=True, title="Dimensión X"), 
                yaxis=dict(showticklabels=True, title="Dimensión Y"), 
                zaxis=dict(showticklabels=True, title="Dimensión Z")
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_q, use_container_width=True)
        
        if st.button("🗑️ Limpiar Historial de Trayectoria", use_container_width=True):
            st.session_state.query_trace_data = []
            save_logs(); st.rerun()
            
        st.markdown(f"""
        <div class="explanation-box">
            <h4>🎯 Auditoría y Trazabilidad Geométrica (RAG Shield)</h4>
            <p>Esta visualización es una <b>herramienta de auditoría de precisión</b> que muestra en tiempo real cómo las consultas del usuario interactúan con el universo de datos de <b>Brain Balance</b>.</p>
            <hr style="opacity: 0.1; margin: 10px 0;">
            <p><b>Significado de la Codificación Visual:</b></p>
            <ul>
                <li><span style="color:{PALETTE['hit']}; font-weight: bold;">● Esferas Verdes (HIT):</span> Éxito. El motor RAG localizó contexto relevante en los manuales y la IA generó una respuesta factual y útil.</li>
                <li><span style="color:{PALETTE['miss']}; font-weight: bold;">● Esferas Rojas (MISS):</span> Alerta de Vacío. La consulta aterrizó en un "desierto semántico". Ocurre cuando el usuario pregunta por temas no documentados o la IA detecta que no tiene información suficiente para ser veraz.</li>
            </ul>
            <p><b>Valor Estratégico de esta Información:</b></p>
            <ol>
                <li><b>Identificar Vacíos (Knowledge Gaps):</b> Una acumulación de esferas rojas es una señal directa de que faltan manuales o capacitación en esa área específica.</li>
                <li><b>Validación de Relevancia:</b> Permite comprobar visualmente si una pregunta de <i>Seguridad</i> aterriza correctamente en el clúster de documentos de <i>Seguridad</i>.</li>
                <li><b>Transparencia Corporativa:</b> Desmitifica el funcionamiento de la IA, demostrando que cada respuesta tiene un fundamento geográfico real dentro de los datos de la empresa.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else: 
        st.info("Realice una consulta para visualizar su trayectoria en el espacio vectorial.")

# --- TAB 4: DOCS ---
with tab_docs:
    st.markdown(f"### {ICON_DATA} Gestión de Chunks e Ingeniería de Datos", unsafe_allow_html=True)
    if "chunks" in st.session_state:
        df_c = [{"ID": f"#{i}", "Cat": c.metadata.get("doc_type").upper(), "File": os.path.basename(c.metadata.get("source", "N/A")), "Snippet": c.page_content[:250]+"..."} for i, c in enumerate(st.session_state.chunks)]
        st.dataframe(df_c, use_container_width=True, height=400)
        
        st.markdown("""
        <div class="explanation-box">
            <h4>🧩 Fase 3: Ingeniería de Contexto (Context Window)</h4>
            <p>No podemos enviar todos los manuales a la memoria de la IA a la vez. Dividimos los documentos en piezas de 500 caracteres.</p>
            <div class="code-snippet">
# LangChain: Troceado de documentos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=120)
chunks = splitter.split_documents(all_docs)
            </div>
            <p><b>Overlap (Pegamento Semántico):</b> El solapamiento de 120 caracteres asegura que si una idea se corta a la mitad, el siguiente fragmento mantenga el contexto, evitando la <b>"Amnesia de Bordes"</b>.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 5: STACK ---
with tab_stack:
    st.markdown(f"### {ICON_LAYERS} Infraestructura Brain Balance", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("<div class='info-card'><h4>🧠 Cerebro Lógico</h4><div class='tech-pill'>LangChain Classic</div><div class='tech-pill'>GPT-4o-mini</div><p>Motor de razonamiento y orquestación de flujos RAG.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'><h4>📦 Almacén Vectorial</h4><div class='tech-pill'>ChromaDB Native</div><div class='tech-pill'>JSON Dynamic Log</div><p>Memoria semántica persistente y log de trazabilidad.</p></div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='info-card'><h4>🪐 Geometría Cuántica</h4><div class='tech-pill'>OpenAI Embeddings v3</div><div class='tech-pill'>Scikit-Learn (t-SNE)</div><p>Reducción de dimensionalidad para auditoría visual en 3D.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='info-card'><h4>📊 Frontend Engine</h4><div class='tech-pill'>Streamlit Framework</div><div class='tech-pill'>Plotly Dynamics</div><p>Interfaz reactiva de alta fidelidad para presentaciones.</p></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; padding: 50px; color: #94a3b8; font-size: 0.85rem;'>Brain Balance Showcase • Intelligence Dashboard • 2024</div>", unsafe_allow_html=True)
