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
  9. Supports both Groq (LLaMA) and Google Gemini models
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
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .study-badge { background: #1f77b4; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
    .metric-card { background: #f0f2f6; padding: 10px; border-radius: 8px; margin: 4px; }
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


_init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎓 StudyMate AI")
    st.caption("Agentic Academic Research & Study Planning Assistant")

    st.divider()

    # ── LLM Provider Selection ─────────────────────────────────────────────────
    st.subheader("🤖 AI Model")
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "gemini"],
        index=0,
        help="Groq = free LLaMA models | Gemini = Google's AI model",
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

        # GPT-OSS models are OpenAI's open-weight models hosted on Groq's
        # LPU hardware — these are the current production model IDs on Groq.
        groq_model = st.selectbox(
            "Groq Model",
            [
                "openai/gpt-oss-120b",   # flagship — 120B params, ~500 t/s
                "openai/gpt-oss-20b",    # faster/cheaper — 20B params, ~1000 t/s
            ],
            index=0,
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

        # Confirmed stable Gemini 3.x Flash endpoints (Google AI docs, Aug 2026)
        gemini_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-3.7-flash",   # latest — best coding & agentic workflows
                "gemini-3.6-flash",   # previous gen — fast multimodal
                "gemini-3.5-flash",   # legacy — high-throughput baseline
            ],
            index=0,
        )
        Config.GEMINI_MODEL = gemini_model

    if Config.get_active_api_key():
        st.success(f"✅ {get_provider_name()}")

    st.divider()

    # ── Study Preferences ──────────────────────────────────────────────────────
    st.subheader("📚 Study Preferences")

    difficulty = st.select_slider(
        "Your Level",
        options=["beginner", "intermediate", "advanced"],
        value=st.session_state.student_profile.difficulty,
    )
    st.session_state.student_profile.update_difficulty(difficulty)

    st.divider()

    # ── Document Upload ────────────────────────────────────────────────────────
    st.subheader("📄 Upload Study Material")
    uploaded_files = st.file_uploader(
        "PDFs, Notes, Textbooks (PDF / TXT / DOCX / MD)",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if st.button("📥 Index Study Material", use_container_width=True) and uploaded_files:
        processor = DocumentProcessor()
        total_chunks = 0
        with st.spinner("Processing and embedding study material…"):
            for file in uploaded_files:
                if file.name not in st.session_state.docs_loaded:
                    chunks = processor.load_from_bytes(file.read(), file.name)
                    st.session_state.vector_store.add_documents(chunks)
                    total_chunks += len(chunks)
                    st.session_state.docs_loaded.append(file.name)
                    st.session_state.student_profile.add_topic_studied(file.name)
        st.success(f"✅ Indexed {total_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.docs_loaded:
        st.markdown("**Loaded material:**")
        for name in st.session_state.docs_loaded:
            st.markdown(f"- 📄 {name}")

    st.divider()

    # ── Student Progress ───────────────────────────────────────────────────────
    st.subheader("📊 Your Progress")
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
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────

st.title("🎓 StudyMate AI")
st.caption(
    "Powered by LangGraph · LangChain · Groq LLaMA / Google Gemini · FAISS · sentence-transformers"
)

# ── Quick Action Buttons ──────────────────────────────────────────────────────
st.markdown("**Quick Actions:**")
qa_col, ex_col, quiz_col, plan_col, eval_col = st.columns(5)

# FIX: initialise quick_query BEFORE the button blocks so it is always
# defined by the time we reach `if quick_query and not user_query` below.
# The original code only set it inside button if-blocks, causing a NameError
# if the page loaded without any button being pressed.
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
    st.subheader("💬 Study Chat")

    # Render chat history
    for role, content, task in st.session_state.chat_history:
        with st.chat_message(role):
            if task and role == "assistant":
                task_icons = {
                    "qa": "❓", "explain": "💡", "quiz": "📝",
                    "plan": "📅", "evaluate": "✅"
                }
                st.caption(f"{task_icons.get(task, '🤖')} {task.upper()} Mode")
            st.markdown(content)

    # HITL Checkpoint — show approval UI if a plan is pending
    if st.session_state.hitl_pending:
        st.warning("⚠️ **Human Approval Required** — Please review the study plan above.")
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
            st.error("⛔ Please enter your API key in the sidebar.")
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

                        # FIX: HITL detection — check hitl_approved key is
                        # present AND explicitly False (not just falsy/absent)
                        if (
                            node_name == "plan_agent"
                            and "hitl_approved" in updates
                            and updates["hitl_approved"] is False
                        ):
                            st.session_state.hitl_pending = True

        except Exception as e:
            st.error(f"⛔ Agent error: {e}")
            st.stop()

        st.session_state.agent_traces = traces

        # Fallback: if streaming didn't surface a final_answer, pull the last
        # meaningful trace content (strip the **[NODE]** prefix first).
        if not final_answer and traces:
            for trace in reversed(traces):
                # Remove the bold node prefix added above
                content = trace.split("] ", 1)[-1].strip() if "] " in trace else trace
                if content and not content.startswith("**["):
                    final_answer = content
                    break

        if not final_answer:
            final_answer = "I processed your request. Please check the agent trace for details."

        # Display final answer
        with st.chat_message("assistant"):
            task_icons = {"qa": "❓", "explain": "💡", "quiz": "📝", "plan": "📅", "evaluate": "✅"}
            st.caption(f"{task_icons.get(detected_task, '🤖')} {detected_task.upper()} Mode")
            st.markdown(final_answer)

        st.session_state.chat_history.append(("assistant", final_answer, detected_task))

        # Update conversation memory
        st.session_state.memory.add_turn(user_query, final_answer)

        # Update student profile
        profile.add_topic_studied(f"query: {user_query[:50]}")


# ── Trace Column ──────────────────────────────────────────────────────────────

with col_trace:
    st.subheader("🔍 Agent Trace")

    if st.session_state.agent_traces:
        with st.expander("Show last run trace", expanded=True):
            for trace in st.session_state.agent_traces:
                st.markdown(trace)
                st.divider()
    else:
        st.info("Agent steps will appear here after your first query.")

    st.subheader("🧠 Session Memory")
    mem = st.session_state.memory
    st.metric("Conversation Turns", mem.turn_count)

    if mem.turn_count > 0:
        with st.expander("Conversation context"):
            st.text(mem.get_context())

    st.subheader("👤 Student Profile")
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
    "<div style='text-align:center; color:gray; font-size:12px'>"
    "NIELIT Agentic AI Internship Project · StudyMate AI · Multi-Agent Academic Assistant"
    "</div>",
    unsafe_allow_html=True,
)
