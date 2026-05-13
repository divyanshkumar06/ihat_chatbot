"""
╔══════════════════════════════════════════════════════════════════╗
║         UPKSK Swasthya Mitra — Citizen Health Assistant         ║
║   Built with LangChain · ChromaDB · Gemini · Streamlit         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import os
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────── Constants ───────────────────────────
DATA_PATH = Path(__file__).parent / "health_insurance_data.txt"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PROVIDER_OPTIONS = ["Groq (Fast & Free)", "Google Gemini"]

GOOGLE_MODEL_OPTIONS = {
    "gemini-2.0-flash-lite": "⚡ Gemini 2.0 Flash-Lite (fastest)",
    "gemini-2.0-flash": "🚀 Gemini 2.0 Flash (balanced)",
    "gemini-1.5-flash": "💎 Gemini 1.5 Flash (stable)",
}

GROQ_MODEL_OPTIONS = {
    "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B (smart & fast)",
    "llama-3.1-8b-instant": "🦙 Llama 3.1 8B (fastest)",
    "gemma2-9b-it": "✨ Gemma 2 9B (Google open-weights)"
}
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5

# ─────────────────────────── Page Config ─────────────────────────
st.set_page_config(
    page_title="UPKSK Swasthya Mitra",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App Background ── */
.stApp {
    background: linear-gradient(170deg, #060b18 0%, #0a1628 40%, #0d1117 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1225 0%, #081020 100%);
    border-right: 1px solid rgba(34, 197, 94, 0.12);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #b0c8b8 !important;
    font-size: 0.88rem;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #d0e8d8 !important;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* ── Sidebar Buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(34, 197, 94, 0.06) !important;
    border: 1px solid rgba(34, 197, 94, 0.18) !important;
    color: #b0d8c0 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.25s ease;
    text-align: left !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(34, 197, 94, 0.18) !important;
    border-color: rgba(34, 197, 94, 0.4) !important;
    color: #fff !important;
    transform: translateX(4px);
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: rgba(10, 18, 32, 0.75) !important;
    border: 1px solid rgba(34, 197, 94, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    margin-bottom: 0.6rem !important;
    backdrop-filter: blur(12px);
    animation: msgSlide 0.35s ease-out;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.06) 100%) !important;
    border-color: rgba(34, 197, 94, 0.2) !important;
}

@keyframes msgSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #d0dde8 !important;
    font-size: 0.92rem;
    line-height: 1.7;
}
[data-testid="stChatMessage"] strong {
    color: #6ee7a0 !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(34, 197, 94, 0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(10, 18, 32, 0.9) !important;
    border: 1px solid rgba(34, 197, 94, 0.2) !important;
    border-radius: 14px !important;
    color: #e0e8f0 !important;
    font-size: 0.92rem !important;
    padding: 0.9rem 1.2rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(34, 197, 94, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1) !important;
}

/* ── Expander (Sources) ── */
.streamlit-expanderHeader {
    background: rgba(10, 18, 32, 0.5) !important;
    border: 1px solid rgba(34, 197, 94, 0.1) !important;
    border-radius: 10px !important;
    color: #70a880 !important;
    font-size: 0.82rem !important;
}
.streamlit-expanderContent {
    background: rgba(8, 14, 26, 0.8) !important;
    border: 1px solid rgba(34, 197, 94, 0.08) !important;
    border-radius: 0 0 10px 10px !important;
    color: #90b0a0 !important;
    font-size: 0.8rem !important;
}

/* ── Text Inputs ── */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(10, 18, 32, 0.8) !important;
    border: 1px solid rgba(34, 197, 94, 0.2) !important;
    border-radius: 10px !important;
    color: #b0d8c0 !important;
    font-size: 0.85rem !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: rgba(34, 197, 94, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.12) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(34, 197, 94, 0.1) !important;
}

.stAlert {
    background: rgba(10, 18, 32, 0.7) !important;
    border: 1px solid rgba(34, 197, 94, 0.15) !important;
    border-radius: 12px !important;
    color: #b0c8b8 !important;
}

.stSpinner > div { color: #22c55e !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #060b18; }
::-webkit-scrollbar-thumb {
    background: rgba(34, 197, 94, 0.25);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(34, 197, 94, 0.45);
}


@media (max-width: 768px) {
    .hero-section h1 { font-size: 1.6rem !important; }
    .hero-section p  { font-size: 0.88rem !important; }
    [data-testid="stChatMessage"] {
        padding: 0.8rem 1rem !important;
        border-radius: 12px !important;
    }
    .stat-card { padding: 0.7rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════ RAG PIPELINE ════════════════════════

@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    """Load UPKSK hospital data, split into chunks, embed, and
    store in ChromaDB. Cached so it runs only once."""

    # Delete old chroma_db to rebuild with new data
    import shutil
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


@st.cache_resource(show_spinner=False)
def create_chain(_vectorstore, api_key: str, provider: str, model_name: str):
    """Build a ConversationalRetrievalChain with citizen-centric
    health triage prompt and MMR retrieval."""

    if provider == "Google Gemini":
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=1024,
        )
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
        )

    qa_template = """You are **Swasthya Mitra**, a citizen health assistant for the \
Government of Uttar Pradesh's UPKSK (Uttar Pradesh ke Swasthya Kendra) program. \
Your role is to help citizens find the right hospital and specialist doctor based \
on their medical symptoms and location.

### Your Approach:
1. **Understand the symptoms** — Read the citizen's description carefully.
2. **Ask follow-up questions** if the symptoms are vague or could point to multiple \
   specialties. Ask only 1-2 focused questions at a time. Examples:
   - "Is the pain in a bone/joint or in the muscles?"
   - "Did this happen after an injury or gradually?"
   - "How long have you had this symptom?"
3. **Identify the right specialist** — Map symptoms to the correct medical department \
   (Orthopedics, Cardiology, Neurology, etc.)
4. **Recommend specific hospitals** — From the context, recommend hospitals where \
   that specialist is available. Include hospital name, address, OPD timings, and \
   doctor names when available.
5. **Mention emergency protocols** — If symptoms suggest an emergency (chest pain, \
   stroke, severe trauma), IMMEDIATELY advise calling 108 ambulance.

### Rules:
1. Answer ONLY from the provided context. NEVER invent hospital names or doctor names.
2. If you cannot find a matching hospital in context, say: \
   "I don't have this specific hospital information in my database. \
    Please call the UPKSK helpline 1800-180-5145 (toll-free) for assistance."
3. Be empathetic and use simple language — many citizens may not know medical terms.
4. Use **bold** for hospital names, doctor names, and important details.
5. Always mention if the hospital accepts **Ayushman Bharat** card.
6. When suggesting hospitals, prefer ones in the citizen's mentioned district/city.
7. **LANGUAGE RULE:** ALWAYS reply in the exact same language the user uses!
   - If the user types in pure Hindi (Devanagari), reply in pure Hindi.
   - If the user types in Hinglish (Hindi written in English alphabet, e.g., "mera pair dard kar raha hai"), reply in Hinglish.
   - If the user types in English, reply in English.
8. For emergencies, always start with: "🚨 **This sounds like an emergency!** Call \
   108 for an ambulance immediately."

### Context:
{context}

### Citizen's Question:
{question}

### Response:"""

    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )

    condense_template = """Given the chat history and a follow-up question below, \
rephrase the follow-up into a fully self-contained standalone question that \
captures the citizen's complete medical concern and location. Do NOT answer.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""

    CONDENSE_PROMPT = PromptTemplate(
        template=condense_template,
        input_variables=["chat_history", "question"],
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7},
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_PROMPT,
        verbose=False,
    )
    return chain

def analyze_image(image_bytes, provider, api_key):
    """Send image to a Vision-Language Model to analyze wounds."""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = """Analyze this image of an injury or medical condition.
1. Describe what you see (e.g., cut, burn, swelling, rash).
2. Suggest immediate first aid if applicable.
3. Identify the likely medical specialist needed (e.g., General Surgeon, Dermatologist, Orthopedist) and recommend visiting a UPKSK government hospital.
Keep it brief, empathetic, and professional. Always advise consulting a doctor for proper medical advice."""

    if provider == "Google Gemini":
        if not api_key:
            raise ValueError("Please provide a Google API Key in the sidebar.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        msg = llm.invoke([{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}])
        return msg.content
    else:
        if not api_key:
            raise ValueError("Please provide a Groq API Key in the sidebar.")
        from langchain_groq import ChatGroq
        # Groq's new active multimodal model for image input
        llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=api_key)
        msg = llm.invoke([{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}])
        return msg.content

# ════════════════════════ SESSION STATE INIT ══════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ═══════════════════════════ SIDEBAR ═════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.2rem 0 0.6rem;">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;">🏥</div>
        <h2 style="margin:0; color:#d0f0d8; font-weight:700;
                   font-size:1.25rem; letter-spacing:-0.02em;">
            Swasthya Mitra
        </h2>
        <p style="margin:0.25rem 0 0; color:#5a8868; font-size:0.72rem;
                  font-weight:500; letter-spacing:0.05em;">
            UPKSK CITIZEN HEALTH ASSISTANT
        </p>
        <p style="margin:0.3rem 0 0; color:#3d6048; font-size:0.68rem;
                  font-weight:400;">
            उत्तर प्रदेश के स्वास्थ्य केंद्र
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    selected_provider = st.selectbox("🌐 Chat AI Provider", options=PROVIDER_OPTIONS, index=0)
    
    st.markdown("##### 🔑 API Key")

    if selected_provider == "Google Gemini":
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Paste Gemini API key…",
        )
        api_key = api_key_input or os.getenv("GOOGLE_API_KEY", "")

        selected_model = st.selectbox(
            "🤖 Text AI Model",
            options=list(GOOGLE_MODEL_OPTIONS.keys()),
            format_func=lambda x: GOOGLE_MODEL_OPTIONS[x],
            index=0,
            help="Switch models if one is rate-limited.",
        )
    else:
        api_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq API key…",
        )
        api_key = api_key_input or os.getenv("GROQ_API_KEY", "")

        selected_model = st.selectbox(
            "🤖 Text AI Model",
            options=list(GROQ_MODEL_OPTIONS.keys()),
            format_func=lambda x: GROQ_MODEL_OPTIONS[x],
            index=0,
            help="Switch models if one is rate-limited.",
        )

    st.divider()

    st.markdown("### 💡 Try Asking")
    suggestions = [
        "I have severe leg pain after a fall",
        "My child has high fever since 3 days",
        "Chest pain and breathlessness",
        "Which hospitals in Lucknow have orthopedics?",
        "How do I get Ayushman Bharat card?",
        "I need an eye checkup in Varanasi",
    ]
    for q in suggestions:
        if st.button(q, key=f"sug_{q}", use_container_width=True):
            st.session_state.pending_question = q



    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown("""
    <div style="text-align:center; padding:0.4rem 0;">
        <p style="color:#3d5a48; font-size:0.7rem; margin:0;">
            🚨 Emergency: Dial <b style="color:#ef4444;">108</b>
        </p>
        <p style="color:#3d5a48; font-size:0.68rem; margin:0.2rem 0 0;">
            📞 Helpline: <b>1800-180-5145</b>
        </p>
        <hr style="border-color: rgba(34,197,94,0.08); margin: 0.5rem 0;">
        <p style="color:#2d4a38; font-size:0.65rem; margin:0;">Powered by</p>
        <p style="color:#3d6a48; font-size:0.68rem; font-weight:600; margin:0.15rem 0 0;">
            LangChain · ChromaDB · Gemini
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════ MAIN AREA ═══════════════════════════

st.markdown("""
<div class="hero-section" style="text-align:center; padding:1.8rem 1rem 1.2rem;">
    <h1 style="margin:0; font-size:2.2rem; font-weight:800;
               background:linear-gradient(135deg, #22c55e 0%, #10b981 40%, #06b6d4 100%);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               letter-spacing:-0.03em;">
        Swasthya Mitra
    </h1>
    <p style="margin:0.3rem auto 0; max-width:560px; color:#5a8870;
              font-size:0.92rem; font-weight:400; line-height:1.6;">
        Your UPKSK health assistant — describe your symptoms and I'll help
        you find the right hospital and specialist doctor near you.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
stats = [
    ("🏥", "5 Districts", "Hospitals Indexed"),
    ("🩺", "13 Specialties", "Doctors Mapped"),
    ("🔍", "Smart Triage", "Symptom Analysis"),
    ("💬", "Follow-Ups", "Conversational AI"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], stats):
    col.markdown(f"""
    <div class="stat-card" style="text-align:center; padding:0.9rem 0.5rem;
                background:rgba(10,18,32,0.6); border:1px solid rgba(34,197,94,0.1);
                border-radius:12px; margin-bottom:1rem;">
        <div style="font-size:1.4rem;">{icon}</div>
        <div style="color:#a0d8b0; font-size:0.82rem; font-weight:600;
                    margin-top:0.3rem;">{title}</div>
        <div style="color:#4a7858; font-size:0.7rem;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Gate: require API key ──
if not api_key:
    st.info(
        f"👈 **Enter your {selected_provider} API key** in the sidebar to start chatting."
    )
    st.stop()

# ── Initialize RAG ──
with st.spinner("🔧 Building UPKSK knowledge base (first run only)…"):
    try:
        vectorstore = build_knowledge_base()
    except Exception as e:
        st.error(f"❌ Failed to build knowledge base: {e}")
        st.stop()

try:
    chain = create_chain(vectorstore, api_key, selected_provider, selected_model)
except Exception as e:
    st.error(f"❌ Failed to initialize AI chain: {e}")
    st.stop()

# ── Welcome message ──
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "🙏 **Namaste! I am Swasthya Mitra** — your UPKSK health assistant.\n\n"
            "I can help you with:\n\n"
            "- 🩺 **Find the right specialist** — Describe your symptoms and I'll "
            "identify which doctor you need\n"
            "- 🏥 **Locate hospitals** — Find government hospitals near you with "
            "the specialist you need\n"
            "- ⏰ **OPD timings & doctor names** — Know when to visit and who to see\n"
            "- 🆓 **Government schemes** — Ayushman Bharat, Janani Suraksha & more\n"
            "- 🚨 **Emergency guidance** — What to do in medical emergencies\n\n"
            "**Tell me your problem or symptoms, and which city/district you're in!** 💬"
        ),
    })

# ── Render Chat History ──
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 View Retrieved Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {i}:** {src[:200]}…")

# ── Handle Input ──
prompt_obj = st.chat_input(
    "Describe your symptoms or upload a photo...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
    accept_audio=True
)

user_input = None
voice_audio = None
vision_image = None

if prompt_obj:
    if hasattr(prompt_obj, "text") and prompt_obj.text:
        user_input = prompt_obj.text
    elif isinstance(prompt_obj, dict) and prompt_obj.get("text"):
        user_input = prompt_obj.get("text")
        
    if hasattr(prompt_obj, "audio") and prompt_obj.audio:
        voice_audio = prompt_obj.audio
    elif isinstance(prompt_obj, dict) and prompt_obj.get("audio"):
        voice_audio = prompt_obj.get("audio")
        
    if hasattr(prompt_obj, "files") and prompt_obj.files:
        vision_image = prompt_obj.files[0]
    elif isinstance(prompt_obj, dict) and prompt_obj.get("files"):
        vision_image = prompt_obj.get("files")[0]

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

# ── Process Voice Input if present ──
if voice_audio and voice_audio.name not in st.session_state.get("processed_audio", set()):
    if "processed_audio" not in st.session_state:
        st.session_state.processed_audio = set()
    st.session_state.processed_audio.add(voice_audio.name)
    
    with st.spinner("🎙️ Transcribing voice..."):
        try:
            from groq import Groq
            # Fallback to the default key if they are currently on Google Gemini
            groq_key = api_key if "Groq" in selected_provider else os.getenv("GROQ_API_KEY", "")
            
            if not groq_key:
                raise ValueError("Groq API Key is required for Voice Input. Please enter it in the sidebar.")
                
            client = Groq(api_key=groq_key)
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", voice_audio.getvalue()),
                model="whisper-large-v3-turbo",
            )
            user_input = transcription.text
        except Exception as e:
            st.error(f"⚠️ Voice Transcription Error: {e}")

# ── Process Image if present ──
if vision_image and vision_image.name not in st.session_state.get("processed_images", set()):
    if "processed_images" not in st.session_state:
        st.session_state.processed_images = set()
    st.session_state.processed_images.add(vision_image.name)
    
    with st.chat_message("user", avatar="👤"):
        st.image(vision_image, width=300)
        st.markdown("📸 *Uploaded an image for analysis.*")
        
    st.session_state.messages.append({"role": "user", "content": "📸 *Uploaded an image for analysis.*"})
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing wound/injury image…"):
            try:
                analysis = analyze_image(vision_image.getvalue(), selected_provider, api_key)
                st.markdown(analysis)
                st.session_state.messages.append({"role": "assistant", "content": analysis})
                st.session_state.chat_history.append(("I uploaded a photo of my injury.", analysis))
            except Exception as e:
                st.error(f"⚠️ Vision model error: {e}")
                
    st.session_state.pending_image = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing your symptoms…"):
            result = None
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    result = chain.invoke({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history,
                    })
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        if attempt < MAX_RETRIES - 1:
                            wait = RETRY_BASE_DELAY * (2 ** attempt)
                            st.toast(f"⏳ Rate limited — retrying in {wait}s…")
                            time.sleep(wait)
                            continue
                    break

            if result is not None:
                answer = result.get("answer", "I'm sorry, I couldn't generate a response.")
                source_docs = result.get("source_documents", [])
                sources = [doc.page_content for doc in source_docs]

                st.markdown(answer)

                if sources:
                    with st.expander("📎 View Retrieved Sources"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"**Chunk {i}:** {src[:250]}…")

                st.session_state.chat_history.append((user_input, answer))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            else:
                error_str = str(last_error)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    error_msg = (
                        "⚠️ **Quota exhausted** for the current model "
                        f"(`{selected_model}`).\n\n"
                        "**Fix:** Switch to a different model in the sidebar — "
                        "each model has its own separate free-tier quota."
                    )
                else:
                    error_msg = f"⚠️ An error occurred: `{last_error}`"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
