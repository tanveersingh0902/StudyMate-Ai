# 🎓 StudyMate AI
### NIELIT Agentic AI Internship Project

A production-ready **multi-agent AI system** that acts as a personalised academic research and study planning assistant, using **LangGraph**, **RAG**, and your choice of **Groq (LLaMA)** or **Google Gemini**.

---

## 🏗️ Architecture

```
Student Query
     │
     ▼
┌─────────────┐
│ TASK ROUTER │  Detects: QA / Quiz / Plan / Explain / Evaluate
└──────┬──────┘
       │
  ┌────┴───────────────────────────────────────────────────────┐
  ▼                 ▼           ▼            ▼          ▼
PLANNER        (direct)    (direct)      (direct)   (direct)
  │
  ▼
RETRIEVER (FAISS vector search on your uploaded study material)
  │
  ├─► QA AGENT       → answers factual questions
  ├─► EXPLAIN AGENT  → deep structured explanations
  ├─► QUIZ AGENT     → generates MCQ quizzes in JSON
  ├─► PLAN AGENT     → personalised study schedules (HITL!)
  └─► EVALUATE AGENT → marks your answers and gives feedback
         │
         ▼
      CRITIC         → validates QA output (loops back if FAIL)
         │
         ▼
    SYNTHESIZER      → polishes the final response
         │
         ▼
        END
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| Language Models | Groq LLaMA-3.3-70B **or** Google Gemini-2.0-Flash |
| Embeddings | HuggingFace sentence-transformers (local, free) |
| Vector Store | FAISS |
| UI | Streamlit |
| Memory | Rolling-window ConversationMemory + StudentProfile |

---

## ⚙️ Setup

### 1. Clone the project
```bash
git clone https://github.com/YOUR_USERNAME/studymate-ai.git
cd studymate-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env — add your GROQ_API_KEY or GEMINI_API_KEY
```

**Groq (free):** https://console.groq.com  
**Gemini (free):** https://aistudio.google.com/app/apikey

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🎯 What Can StudyMate Do?

| You type | What happens |
|---|---|
| "What is photosynthesis?" | QA Agent answers from your uploaded notes |
| "Explain Newton's laws for a beginner" | Explain Agent gives structured explanation |
| "Generate a quiz on Chapter 3" | Quiz Agent creates 5 MCQs |
| "Make me a 7-day study plan" | Plan Agent creates schedule + **asks approval** |
| "Evaluate my answer: F = ma..." | Evaluate Agent scores and gives feedback |

---

## 🚀 Deployment

### Option 1: Streamlit Community Cloud (Easiest)

1. Push your code to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set main file to `app.py`
4. In **Advanced settings → Secrets**, add your API keys:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   # or
   GEMINI_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy** 🎉

### Option 2: Docker

```bash
# Build
docker build -t studymate-ai .

# Run (pass API key as env var)
docker run -p 8501:8501 \
  -e GROQ_API_KEY="gsk_your_key_here" \
  studymate-ai
```

Then open http://localhost:8501

### Option 3: Render / Railway

1. Connect your GitHub repo
2. Set **Build Command**: `pip install -r requirements.txt`
3. Set **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Add environment variables: `GROQ_API_KEY` or `GEMINI_API_KEY`

### Option 4: Heroku

```bash
heroku create studymate-ai
heroku config:set GROQ_API_KEY="gsk_your_key_here"
git push heroku main
```

---

## 📂 Project Structure

```
studymate-ai/
├── app.py                    # Streamlit UI
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template (safe to commit)
├── .streamlit/
│   └── config.toml           # Streamlit server config
├── Dockerfile                # Docker deployment
├── Procfile                  # Heroku/Railway deployment
├── runtime.txt               # Python version for PaaS
├── packages.txt              # System deps for Streamlit Cloud
├── README.md
└── src/
    ├── __init__.py
    ├── config.py             # Config for Groq + Gemini + all settings
    ├── llm_factory.py        # Unified LLM factory (Groq / Gemini)
    ├── document_processor.py # File loading & chunking
    ├── vector_store.py       # FAISS + HuggingFace embeddings
    ├── graph.py              # LangGraph 8-agent workflow
    └── memory.py             # ConversationMemory + StudentProfile
```

---

## 🔑 API Keys

| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| **Groq** | ✅ Generous free tier | [console.groq.com](https://console.groq.com) |
| **Gemini** | ✅ Free tier available | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

You only need **one** provider — pick whichever you prefer. You can switch between them in the sidebar at runtime.

---

<div style='text-align:center; color:gray; font-size:12px'>
NIELIT Agentic AI Internship Project · StudyMate AI · Multi-Agent Academic Assistant
</div>
