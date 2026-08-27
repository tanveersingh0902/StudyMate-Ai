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
| Language Models | Groq LLaMA-3.3-70B **or** Google Gemini-1.5-Flash |
| Embeddings | HuggingFace sentence-transformers (local, free) |
| Vector Store | FAISS |
| UI | Streamlit |
| Memory | Rolling-window ConversationMemory + StudentProfile |

---

## ⚙️ Setup

### 1. Clone / download the project
```bash
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



---

## 📂 Project Structure

```
studymate-ai/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
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
