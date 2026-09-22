"""
app.py — Streamlit UI for StudyMate AI: Agentic Academic Research & Study Planning Assistant.

Run with:
    streamlit run app.py

Features:
  1. Upload study material (PDF / TXT / DOCX / MD)
  2. Ask questions from your study material (RAG-powered QA)
  3. Get explanations at different difficulty levels
  4. Generate personalised study plans
  5. Take auto-generated quizzes
  6. Submit answers for AI evaluation
  7. Tracks your learning progress across the session
  8. Human-in-the-Loop approval for study plans
  9. Supports both Groq (GPT-OSS) and Google Gemini models
  10. Streams agent responses live

"""

import streamlit as st
from langchain_core.messages import HumanMessage

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.graph import build_graph, StudyState
from src.memory import ConversationMemory, StudentProfile
from src.llm_factory import get_provider_name


# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StudyMate AI — Smart Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark-Mode CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Import Premium Font ───────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Resets ─────────────────────────────────────────── */
*, *::before, *::after { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* ── Glassmorphism Card ────────────────────────────────────── */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 16px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.25);
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.08);
}

/* ── Gradient Header ───────────────────────────────────────── */
.gradient-header {
    background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
    line-height: 1.2;
}

.gradient-sub {
    background: linear-gradient(90deg, #a78bfa 0%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ── Sidebar Styling ───────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 30, 0.98) 0%, rgba(10, 10, 25, 0.99) 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
    border-color: rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
    transition: border-color 0.3s ease !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    border-color: rgba(139, 92, 246, 0.6) !important;
}

/* ── Quick Action Buttons ──────────────────────────────────── */
div.stButton > button {
    background: rgba(139, 92, 246, 0.08) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 10px 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.2px !important;
}
div.stButton > button:hover {
    background: rgba(139, 92, 246, 0.2) !important;
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.15), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    transform: translateY(-2px) !important;
    color: #f8fafc !important;
}
div.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.1) !important;
}

/* Primary button variant */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%) !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.3), 0 4px 15px rgba(0, 0, 0, 0.4) !important;
}

/* ── Chat Messages ─────────────────────────────────────────── */
.stChatMessage {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    transition: border-color 0.3s ease !important;
}
.stChatMessage:hover {
    border-color: rgba(139, 92, 246, 0.15) !important;
}

/* ── Chat Input ────────────────────────────────────────────── */
.stChatInput > div {
    border-radius: 14px !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.stChatInput > div:focus-within {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 15px rgba(139, 92, 246, 0.1) !important;
}

/* ── Metric Cards ──────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(139, 92, 246, 0.06);
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 12px;
    padding: 12px 16px;
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(139, 92, 246, 0.3);
    background: rgba(139, 92, 246, 0.1);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-weight: 700 !important;
}

/* ── Expander ──────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.streamlit-expanderHeader:hover {
    background: rgba(139, 92, 246, 0.08) !important;
    border-color: rgba(139, 92, 246, 0.2) !important;
}

/* ── Slider ────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4) !important;
}

/* ── File Uploader ─────────────────────────────────────────── */
section[data-testid="stFileUploader"] {
    border: 1px dashed rgba(139, 92, 246, 0.3) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease !important;
}
section[data-testid="stFileUploader"]:hover {
    border-color: rgba(139, 92, 246, 0.6) !important;
}

/* ── Dividers ──────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.2), transparent) !important;
    margin: 20px 0 !important;
}

/* ── Status Badges ─────────────────────────────────────────── */
.task-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.task-badge-qa       { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.task-badge-explain  { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.3); }
.task-badge-quiz     { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.task-badge-plan     { background: rgba(139,92,246,0.15); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }
.task-badge-evaluate { background: rgba(236,72,153,0.15); color: #f9a8d4; border: 1px solid rgba(236,72,153,0.3); }

/* ── Trace Steps ───────────────────────────────────────────── */
.trace-step {
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid rgba(139, 92, 246, 0.4);
    padding: 10px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    transition: all 0.3s ease;
}
.trace-step:hover {
    background: rgba(139, 92, 246, 0.06);
    border-left-color: rgba(139, 92, 246, 0.8);
}

/* ── Progress Indicator ────────────────────────────────────── */
.doc-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #6ee7b7;
    margin: 2px 4px 2px 0;
}

/* ── Footer ────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    padding: 24px 0 12px;
    border-top: 1px solid rgba(139, 92, 246, 0.1);
    margin-top: 32px;
    letter-spacing: 0.3px;
}
.app-footer a {
    color: #8b5cf6;
    text-decoration: none;
}

/* ── Section Headers ───────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.5); }

/* ── Success/Warning/Error Alerts ──────────────────────────── */
.stAlert > div {
    border-radius: 12px !important;
    border: none !important;
}

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 500 !important;
}

/* ── Pulse Animation for Loading ───────────────────────────── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.2); }
    50%      { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
}
.loading-active {
    animation: pulse-glow 2s infinite;
}

/* ── Force Dark Mode / Remove Light Mode Toggle ──────── */
:root { color-scheme: dark !important; }

/* Hide the theme‑settings section inside the Settings dialog */
div[data-testid="stThemeSettings"],
div[data-testid="stTheme"],
[data-testid="stAppViewBlockContainer"] .stThemeSettings,
div[class*="ThemeSettings"],
section[data-testid="stSidebarUserContent"] div[data-testid="stTheme"] {
    display: none !important;
}

/* Also hide the Streamlit light/dark toggle radio buttons in settings */
div[data-baseweb="radio"] label:has(span:is([data-testid="stThemeLight"], [data-testid="stMarkdownContainer"])) {
    display: none !important;
}

/* ── Hide Streamlit Defaults ───────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────

def _init_session():
    """Initialise persistent objects in Streamlit session state."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=5)
    if "student_profile" not in st.session_state:
        st.session_state.student_profile = StudentProfile()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_traces" not in st.session_state:
        st.session_state.agent_traces = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = []
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = False
    if "pending_plan" not in st.session_state:
        st.session_state.pending_plan = ""
    if "last_task_type" not in st.session_state:
        st.session_state.last_task_type = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False


_init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Sidebar header
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 4px;">
        <div class="gradient-header" style="font-size:1.6rem;">🎓 StudyMate AI</div>
        <div class="gradient-sub">Agentic Academic Research & Study Planning</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── LLM Provider Selection ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 AI Model</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "gemini"],
        index=0,
        help="Groq = GPT-OSS models on Groq LPUs | Gemini = Google's latest AI models",
    )
    Config.LLM_PROVIDER = provider

    if provider == "groq":
        api_key = st.text_input(
            "Groq API Key",
            value=Config.GROQ_API_KEY,
            type="password",
            help="Free key at https://console.groq.com",
        )
        if api_key:
            Config.GROQ_API_KEY = api_key

        groq_model = st.selectbox(
            "Groq Model",
            [
                "openai/gpt-oss-120b",   # flagship — 120B params, ~500 t/s
                "openai/gpt-oss-20b",    # faster/cheaper — 20B params, ~1000 t/s
            ],
            index=0,
            help="GPT-OSS-120B = flagship quality | GPT-OSS-20B = faster responses",
        )
        Config.GROQ_MODEL = groq_model
        Config.LLM_MODEL = groq_model

    else:  # Gemini
        api_key = st.text_input(
            "Gemini API Key",
            value=Config.GEMINI_API_KEY,
            type="password",
            help="Free key at https://aistudio.google.com/app/apikey",
        )
        if api_key:
            Config.GEMINI_API_KEY = api_key

        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-3.7-flash",   # latest — best coding & agentic workflows
                "gemini-3.6-flash",   # previous gen — fast multimodal
                "gemini-3.5-flash",   # legacy — high-throughput baseline
            ],
            index=0,
            help="Gemini 3.7 Flash = latest & best | 3.6/3.5 = previous generations",
        )
        Config.GEMINI_MODEL = gemini_model

    if Config.get_active_api_key():
        st.success(f"✅ {get_provider_name()}")
    else:
        st.warning("⚠️ Enter your API key to get started")
        st.error(
            f"⚠️ No {provider.upper()} API key found. "
            f"Add it to Streamlit secrets or your .env file."
        )

    st.divider()

    # ── Study Preferences ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Study Preferences</div>', unsafe_allow_html=True)

    difficulty = st.select_slider(
        "Your Level",
        options=["beginner", "intermediate", "advanced"],
        value=st.session_state.student_profile.difficulty,
    )
    st.session_state.student_profile.update_difficulty(difficulty)

    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📄 Upload Study Material</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDFs, Notes, Textbooks (PDF / TXT / DOCX / MD)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Study Material", use_container_width=True) and uploaded_files:
        processor = DocumentProcessor()
        total_chunks = 0
        new_files = 0
        with st.spinner("Processing and embedding study material…"):
            for file in uploaded_files:
                if file.name not in st.session_state.docs_loaded:
                    try:
                        chunks = processor.load_from_bytes(file.read(), file.name)
                        st.session_state.vector_store.add_documents(chunks)
                        total_chunks += len(chunks)
                        new_files += 1
                        st.session_state.docs_loaded.append(file.name)
                        st.session_state.student_profile.add_topic_studied(file.name)
                    except Exception as e:
                        st.error(f"❌ Failed to process {file.name}: {e}")
        if new_files > 0:
            st.success(f"✅ Indexed **{total_chunks}** chunks from **{new_files}** new file(s).")
        elif uploaded_files:
            st.info("ℹ️ All files already indexed.")

    if st.session_state.docs_loaded:
        st.markdown("**Loaded material:**")
        for name in st.session_state.docs_loaded:
            st.markdown(
                f'<span class="doc-chip">📄 {name}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Student Progress ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Your Progress</div>', unsafe_allow_html=True)
    profile = st.session_state.student_profile
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Topics Studied", len(profile.topics_studied))
        st.metric("Questions", profile.total_questions_answered)
    with col2:
        st.metric("Avg Quiz Score", f"{profile.average_score_pct:.0f}%")
        st.metric("Weak Areas", len(profile.weak_areas))

    if profile.weak_areas:
        st.warning(f"⚠️ Focus areas: {', '.join(profile.weak_areas[:3])}")

    st.divider()

    # ── Advanced Settings ──────────────────────────────────────────────────────
    with st.expander("⚙️ Advanced Settings"):
        Config.MAX_RETRIEVAL_DOCS = st.slider("Chunks per query", 1, 10, Config.MAX_RETRIEVAL_DOCS)
        Config.MAX_ITERATIONS = st.slider("Max critique loops", 1, 5, Config.MAX_ITERATIONS)
        Config.TEMPERATURE = st.slider("Temperature", 0.0, 1.0, Config.TEMPERATURE, step=0.05)
        Config.QUIZ_QUESTION_COUNT = st.slider("Quiz questions", 3, 10, Config.QUIZ_QUESTION_COUNT)
        Config.STUDY_PLAN_DAYS = st.slider("Study plan days", 3, 30, Config.STUDY_PLAN_DAYS)

    # ── Reset ──────────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.memory.clear()
        st.session_state.student_profile = StudentProfile()
        st.session_state.chat_history = []
        st.session_state.agent_traces = []
        st.session_state.docs_loaded = []
        st.session_state.hitl_pending = False
        st.session_state.pending_plan = ""
        st.session_state.processing = False
        Config.reset()
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="margin-bottom: 8px;">
    <div class="gradient-header">🎓 StudyMate AI</div>
    <div class="gradient-sub" style="font-size: 0.9rem;">
        Powered by LangGraph · LangChain · Groq GPT-OSS / Google Gemini · FAISS · sentence-transformers
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Action Buttons ──────────────────────────────────────────────────────
st.markdown("""
<div style="margin: 12px 0 4px;">
    <span style="font-weight: 600; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">
        Quick Actions
    </span>
</div>
""", unsafe_allow_html=True)

qa_col, ex_col, quiz_col, plan_col, eval_col = st.columns(5)

quick_query = None

with qa_col:
    if st.button("❓ Ask a Question", use_container_width=True):
        quick_query = "Can you answer a question from my study material?"
with ex_col:
    if st.button("💡 Explain a Concept", use_container_width=True):
        quick_query = "Please explain the main concept from my uploaded material in detail."
with quiz_col:
    if st.button("📝 Take a Quiz", use_container_width=True):
        quick_query = "Generate a quiz from my uploaded study material."
with plan_col:
    if st.button("📅 Create Study Plan", use_container_width=True):
        quick_query = "Create a personalised study plan for my uploaded topics."
with eval_col:
    if st.button("✅ Evaluate My Answer", use_container_width=True):
        quick_query = "Please evaluate my answer: [type your answer below]"

st.divider()

col_chat, col_trace = st.columns([3, 2])

# ── Chat Column ───────────────────────────────────────────────────────────────

with col_chat:
    st.markdown('<div class="section-header">💬 Study Chat</div>', unsafe_allow_html=True)

    # Render chat history
    for role, content, task in st.session_state.chat_history:
        with st.chat_message(role):
            if task and role == "assistant":
                task_badges = {
                    "qa":       ("❓", "QA", "task-badge-qa"),
                    "explain":  ("💡", "EXPLAIN", "task-badge-explain"),
                    "quiz":     ("📝", "QUIZ", "task-badge-quiz"),
                    "plan":     ("📅", "PLAN", "task-badge-plan"),
                    "evaluate": ("✅", "EVALUATE", "task-badge-evaluate"),
                }
                icon, label, css_class = task_badges.get(task, ("🤖", task.upper(), "task-badge-qa"))
                st.markdown(
                    f'<span class="task-badge {css_class}">{icon} {label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(content)

    # HITL Checkpoint — show approval UI if a plan is pending
    if st.session_state.hitl_pending:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.05);">
            <div style="font-weight: 600; color: #fcd34d; margin-bottom: 8px;">
                ⚠️ Human Approval Required
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem;">
                Please review the study plan above and approve or request changes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            if st.button("✅ Approve Plan", use_container_width=True, type="primary"):
                st.session_state.hitl_pending = False
                approved_msg = "✅ Study plan approved! You can now follow this schedule. Good luck with your studies! 🎯"
                with st.chat_message("assistant"):
                    st.markdown(approved_msg)
                st.session_state.chat_history.append(("assistant", approved_msg, "plan"))
                st.session_state.memory.add_turn("[HITL] Approved study plan", approved_msg)
                st.rerun()
        with h_col2:
            if st.button("✏️ Request Changes", use_container_width=True):
                st.session_state.hitl_pending = False
                st.session_state.chat_history.append(
                    ("assistant", "Sure! Please describe what changes you'd like to the study plan.", "plan")
                )
                st.rerun()

    # Chat input
    user_query = st.chat_input("Ask a study question, request a quiz, or ask for a study plan…")

    # Use quick action if no manual input
    if quick_query and not user_query:
        user_query = quick_query

    if user_query:
        # Validate API key
        if not Config.get_active_api_key():
            st.error("⛔ Please enter your API key in the sidebar to get started.")
            st.error("⛔ API key not configured. Please add your API key to Streamlit secrets or your .env file.")
            st.stop()

        # Validate non-empty query
        if not user_query.strip():
            st.warning("Please enter a question or request.")
            st.stop()

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append(("user", user_query, ""))

        # Handle "approve plan" text input
        if "approve plan" in user_query.lower():
            st.session_state.hitl_pending = False

        # Build initial graph state
        prior_messages = st.session_state.memory.get_messages()
        profile = st.session_state.student_profile

        init_state: StudyState = {
            "messages":          prior_messages + [HumanMessage(content=user_query)],
            "query":             user_query,
            "task_type":         "",
            "research_plan":     [],
            "retrieved_docs":    [],
            "analysis":          "",
            "critique_passed":   False,
            "final_answer":      "",
            "quiz_questions":    "",
            "study_plan":        "",
            "explanation":       "",
            "evaluation_result": "",
            "difficulty":        profile.difficulty,
            "iteration":         0,
            "has_documents":     st.session_state.vector_store.is_ready,
            "hitl_approved":     False,
            "student_profile":   profile.get_summary(),
        }

        # Run graph with live streaming
        traces = []
        final_answer = ""
        detected_task = "qa"

        try:
            graph = build_graph(st.session_state.vector_store)

            with st.spinner("🔄 StudyMate agents working…"):
                for step in graph.stream(init_state, stream_mode="updates"):
                    for node_name, updates in step.items():
                        # Collect agent trace messages
                        for msg in updates.get("messages", []):
                            content = getattr(msg, "content", str(msg))
                            traces.append(f"**[{node_name.upper()}]** {content}")

                        # Capture task type
                        if updates.get("task_type"):
                            detected_task = updates["task_type"]
                            st.session_state.last_task_type = detected_task

                        # Capture final answer from synthesizer
                        if node_name == "synthesizer" and updates.get("final_answer"):
                            final_answer = updates["final_answer"]

                        # HITL detection — check hitl_approved key is
                        # present AND explicitly False
                        if (
                            node_name == "plan_agent"
                            and "hitl_approved" in updates
                            and updates["hitl_approved"] is False
                        ):
                            st.session_state.hitl_pending = True

        except Exception as e:
            st.error(f"⛔ Agent error: {e}")
            # Still save traces so user can debug
            st.session_state.agent_traces = traces
            st.stop()

        st.session_state.agent_traces = traces

        # Fallback: if streaming didn't surface a final_answer, pull the last
        # meaningful trace content
        if not final_answer and traces:
            for trace in reversed(traces):
                content = trace.split("] ", 1)[-1].strip() if "] " in trace else trace
                if content and not content.startswith("**["):
                    final_answer = content
                    break

        if not final_answer:
            final_answer = "I processed your request. Please check the agent trace for details."

        # Display final answer with task badge
        with st.chat_message("assistant"):
            task_badges = {
                "qa":       ("❓", "QA", "task-badge-qa"),
                "explain":  ("💡", "EXPLAIN", "task-badge-explain"),
                "quiz":     ("📝", "QUIZ", "task-badge-quiz"),
                "plan":     ("📅", "PLAN", "task-badge-plan"),
                "evaluate": ("✅", "EVALUATE", "task-badge-evaluate"),
            }
            icon, label, css_class = task_badges.get(detected_task, ("🤖", detected_task.upper(), "task-badge-qa"))
            st.markdown(
                f'<span class="task-badge {css_class}">{icon} {label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(final_answer)

        st.session_state.chat_history.append(("assistant", final_answer, detected_task))

        # Update conversation memory
        st.session_state.memory.add_turn(user_query, final_answer)

        # Update student profile
        profile.add_topic_studied(f"query: {user_query[:50]}")


# ── Trace Column ──────────────────────────────────────────────────────────────

with col_trace:
    st.markdown('<div class="section-header">🔍 Agent Trace</div>', unsafe_allow_html=True)

    if st.session_state.agent_traces:
        with st.expander("Show last run trace", expanded=True):
            for trace in st.session_state.agent_traces:
                st.markdown(
                    f'<div class="trace-step">{trace}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🔬</div>
            <div style="color: #64748b; font-size: 0.85rem;">
                Agent steps will appear here after your first query.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top: 20px;">🧠 Session Memory</div>', unsafe_allow_html=True)

    mem = st.session_state.memory
    st.metric("Conversation Turns", mem.turn_count)

    if mem.turn_count > 0:
        with st.expander("Conversation context"):
            st.text(mem.get_context())

    st.markdown('<div class="section-header" style="margin-top: 20px;">👤 Student Profile</div>', unsafe_allow_html=True)
    p = st.session_state.student_profile
    with st.expander("View profile", expanded=False):
        st.text(p.get_summary())


# ── Task Guide ────────────────────────────────────────────────────────────────

st.divider()
with st.expander("📖 How to use StudyMate AI"):
    st.markdown("""
| You type... | What happens |
|---|---|
| "What is Newton's second law?" | QA Agent answers from your uploaded material |
| "Explain photosynthesis for a beginner" | Explain Agent gives a structured explanation |
| "Generate a quiz on Chapter 3" | Quiz Agent creates MCQs from your notes |
| "Make me a 7-day study plan" | Plan Agent creates a schedule (asks for your approval!) |
| "Evaluate my answer: F = ma where F is force..." | Evaluate Agent scores and gives feedback |

**Tips:**
- Upload your textbooks/notes first for best results
- Adjust your level in the sidebar (beginner → advanced)
- The quiz results update your weak areas automatically
- Study plans require your approval before they're finalised (Human-in-the-Loop!)
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    '🎓 NIELIT Agentic AI Internship Project · <strong>StudyMate AI</strong> · Multi-Agent Academic Assistant'
    '</div>',
    unsafe_allow_html=True,
)
