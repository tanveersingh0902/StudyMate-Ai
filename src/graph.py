"""
graph.py — LangGraph multi-agent workflow for StudyMate AI.

Architecture (8 agents):

  START
    │
    ▼
  ┌──────────────┐
  │  TASK ROUTER │  Decides which task to perform
  └──────┬───────┘
         │
    ┌────┴─────────────────────────────────────────────┐
    │           Task Routing Branches                   │
    ▼                 ▼             ▼          ▼        ▼
  [QA AGENT]   [QUIZ AGENT]  [PLAN AGENT]  [EVAL]  [EXPLAIN]
    │                │             │          │         │
    └────────────────┴─────────────┴──────────┴─────────┘
                     │
                     ▼
              ┌────────────┐
              │ SYNTHESIZER│
              └─────┬──────┘
                    ▼
                  END

"""

import json
import logging
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .config import Config
from .vector_store import VectorStore
from .llm_factory import call_llm

logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class StudyState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    task_type: str                # qa / quiz / plan / explain / evaluate
    research_plan: List[str]
    retrieved_docs: List[str]
    analysis: str
    critique_passed: bool
    final_answer: str
    quiz_questions: str           # JSON string of quiz questions
    study_plan: str
    explanation: str
    evaluation_result: str
    difficulty: str               # beginner / intermediate / advanced
    iteration: int
    has_documents: bool
    hitl_approved: bool           # Human-in-the-Loop flag
    student_profile: str          # serialised summary from StudentProfile


# ── Task Router ───────────────────────────────────────────────────────────────

def task_router_node(state: StudyState) -> dict:
    """
    Task Router Agent:
    Reads the student's query and decides WHICH task to perform.
    """
    system = """You are a task routing agent for a student AI assistant.
Classify the student's request into EXACTLY ONE of these task types:
- qa: student is asking a factual question about study material
- quiz: student wants to test themselves with questions
- plan: student wants a study plan or schedule
- explain: student wants a concept explained in detail
- evaluate: student is submitting an answer for feedback/grading

Respond with ONLY the task type word. No explanation."""

    try:
        task = call_llm(system, f"Classify this student request:\n\n{state['query']}").lower().strip()
    except Exception as e:
        logger.warning(f"Task router LLM call failed: {e}")
        task = "qa"  # Safe fallback

    # Validate and fallback to 'qa' if unrecognised
    valid_tasks = {"qa", "quiz", "plan", "explain", "evaluate"}
    if task not in valid_tasks:
        # Try to extract a valid task from the response (LLM may add extra text)
        for valid in valid_tasks:
            if valid in task:
                task = valid
                break
        else:
            task = "qa"

    return {
        "task_type": task,
        "messages": [AIMessage(content=f"🎯 Task identified: **{task.upper()}**")],
    }


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: StudyState) -> dict:
    """
    Planner Agent:
    Breaks down the query into focused retrieval sub-queries.
    """
    query = state["query"]
    task = state.get("task_type", "qa")

    system = f"""You are a study research planner. The student has a {task} request.
Break it into 2-3 focused sub-queries to search study material.
Output ONLY a numbered list of sub-queries, nothing else.
Example:
1. Definition of photosynthesis
2. Steps in light-dependent reactions
3. Role of chlorophyll in energy capture"""

    try:
        plan_text = call_llm(system, f"Break down this study query:\n\n{query}")
    except Exception as e:
        logger.warning(f"Planner LLM call failed: {e}")
        return {
            "research_plan": [query],
            "messages": [AIMessage(content=f"📋 Using original query for search (planner unavailable).")],
        }

    lines = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in plan_text.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    if not lines:
        lines = [query]

    return {
        "research_plan": lines,
        "messages": [AIMessage(content=f"📋 **Research Plan:**\n{plan_text}")],
    }


# ── Retriever ─────────────────────────────────────────────────────────────────

def retriever_node(state: StudyState, vector_store: VectorStore) -> dict:
    """
    Retriever Agent:
    Searches the uploaded study material for the most relevant passages.
    """
    if not state["has_documents"]:
        return {"retrieved_docs": []}

    plan = state.get("research_plan") or [state["query"]]
    all_docs: List[Document] = []

    for sub_query in plan:
        try:
            docs = vector_store.similarity_search(sub_query, k=Config.MAX_RETRIEVAL_DOCS)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Retrieval failed for sub-query '{sub_query[:50]}': {e}")

    # Deduplicate by first 100 chars of content
    seen, unique_docs = set(), []
    for doc in all_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    formatted = [
        f"[Source: {d.metadata.get('source', '?')} | Chunk {d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in unique_docs
    ]

    return {
        "retrieved_docs": formatted,
        "messages": [AIMessage(content=f"🔍 Retrieved **{len(formatted)}** relevant passages from your study material.")],
    }


# ── QA Agent ──────────────────────────────────────────────────────────────────

def qa_agent_node(state: StudyState) -> dict:
    """
    QA Agent:
    Answers student questions using retrieved study material.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    difficulty = state.get("difficulty", Config.DEFAULT_DIFFICULTY)
    profile = state.get("student_profile", "")

    context = "\n\n---\n\n".join(docs) if docs else "No study material provided."

    system = f"""You are a helpful study tutor. Answer the student's question
using the provided study material context where available.

Difficulty level: {difficulty}
- beginner: Use simple language, many examples, avoid jargon
- intermediate: Standard academic language, some examples
- advanced: Technical depth, assume prior knowledge

Student profile context:
{profile}

If the answer is not in the study material, say so clearly and provide
what general knowledge you can."""

    try:
        answer = call_llm(system, f"Student Question: {query}\n\nStudy Material:\n{context}")
    except Exception as e:
        logger.error(f"QA Agent LLM call failed: {e}")
        answer = f"⚠️ I encountered an error while processing your question. Please try again.\n\nError: {e}"

    return {
        "analysis": answer,
        "messages": [AIMessage(content=answer)],
    }


# ── Explain Agent ─────────────────────────────────────────────────────────────

def explain_agent_node(state: StudyState) -> dict:
    """
    Explain Agent:
    Provides a deep, structured explanation of a concept from study material.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    difficulty = state.get("difficulty", "intermediate")

    context = "\n\n---\n\n".join(docs) if docs else ""

    difficulty_instructions = {
        "beginner": "Use very simple language. Explain like the student is new to this topic. Use analogies from everyday life. Break everything into small steps.",
        "intermediate": "Use clear academic language. Explain concepts with examples. Assume basic familiarity with the subject.",
        "advanced": "Use technical language. Go deep into mechanisms, edge cases, and connections to advanced topics.",
    }

    instruction = difficulty_instructions.get(difficulty, difficulty_instructions["intermediate"])

    system = f"""You are an expert tutor providing a comprehensive explanation.

{instruction}

Structure your explanation:
1. **Core Concept** — What is it?
2. **How it Works** — Step-by-step mechanism
3. **Example** — Concrete real-world example
4. **Key Points to Remember** — Bullet list
5. **Common Misconceptions** — What students often get wrong"""

    try:
        explanation = call_llm(system,
                               f"Explain this concept in detail:\n{query}\n\nFrom this study material:\n{context}")
    except Exception as e:
        logger.error(f"Explain Agent LLM call failed: {e}")
        explanation = f"⚠️ I encountered an error while generating the explanation. Please try again.\n\nError: {e}"

    return {
        "explanation": explanation,
        "analysis": explanation,
        "messages": [AIMessage(content=explanation)],
    }


# ── Quiz Agent ────────────────────────────────────────────────────────────────

def quiz_agent_node(state: StudyState) -> dict:
    """
    Quiz Agent:
    Generates multiple-choice questions from the student's study material.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    difficulty = state.get("difficulty", "intermediate")
    n_questions = Config.QUIZ_QUESTION_COUNT

    context = "\n\n---\n\n".join(docs[:4]) if docs else "No study material uploaded — generate general questions on the topic."

    system = f"""You are a quiz generator for students. Generate {n_questions} multiple-choice
questions from the provided study material at {difficulty} level.

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
  "topic": "topic name here",
  "questions": [
    {{
      "id": 1,
      "question": "Question text here?",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct": "A",
      "explanation": "Why A is correct"
    }}
  ]
}}"""

    try:
        quiz_raw = call_llm(system, f"Generate a quiz about:\n{query}\n\nStudy Material:\n{context}")
    except Exception as e:
        logger.error(f"Quiz Agent LLM call failed: {e}")
        return {
            "quiz_questions": "{}",
            "analysis": f"⚠️ I couldn't generate the quiz due to an error. Please try again.\n\nError: {e}",
            "messages": [AIMessage(content=f"⚠️ Quiz generation failed: {e}")],
        }

    # Strip markdown fences the model might add
    quiz_clean = quiz_raw.replace("```json", "").replace("```", "").strip()

    # Try to parse JSON; on failure preserve the raw text so the student sees something
    try:
        quiz_data = json.loads(quiz_clean)
        display = _format_quiz_display(quiz_data)
        stored_json = quiz_clean
    except json.JSONDecodeError:
        display = (
            "⚠️ The quiz couldn't be formatted automatically. Here are the raw questions:\n\n"
            + quiz_raw
        )
        stored_json = quiz_raw  # Preserve raw text instead of empty JSON

    return {
        "quiz_questions": stored_json,
        "analysis": display,
        "messages": [AIMessage(content=display)],
    }


def _format_quiz_display(quiz_data: dict) -> str:
    """Convert quiz JSON into a readable markdown string."""
    lines = [f"## 📝 Quiz: {quiz_data.get('topic', 'Study Quiz')}\n"]
    for i, q in enumerate(quiz_data.get("questions", []), 1):
        # Guard against missing keys to avoid KeyError crashes
        question_text = q.get("question", f"Question {i}")
        lines.append(f"**Q{i}. {question_text}**")
        for letter, text in q.get("options", {}).items():
            lines.append(f"  - {letter}) {text}")
        correct = q.get("correct", "?")
        explanation = q.get("explanation", "")
        lines.append(f"\n> ✅ Answer: **{correct}** — {explanation}\n")
    return "\n".join(lines)


# ── Study Plan Agent ──────────────────────────────────────────────────────────

def plan_agent_node(state: StudyState) -> dict:
    """
    Study Plan Agent:
    Creates a personalised study schedule. Flags for Human-in-the-Loop approval.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    profile = state.get("student_profile", "")
    days = Config.STUDY_PLAN_DAYS
    difficulty = state.get("difficulty", "intermediate")

    topics_from_docs = list(set(
        d.split("|")[0].replace("[Source:", "").strip()
        for d in docs if "[Source:" in d
    )) if docs else ["uploaded material"]

    system = f"""You are a personalised study planner for a student.
Create a {days}-day study plan based on their needs and learning profile.

Student Profile:
{profile}

Available study material topics: {', '.join(topics_from_docs)}
Difficulty Level: {difficulty}

Structure the plan as:
- Daily breakdown (Day 1, Day 2, …)
- Each day: topics to study, suggested activities, time estimate
- Include revision days
- Highlight weak areas that need extra attention
- End with exam preparation tips"""

    try:
        plan = call_llm(system,
                        f"Create a study plan for:\n{query}\n\nMaterial available:\n{chr(10).join(docs[:2]) if docs else 'General topics'}")
    except Exception as e:
        logger.error(f"Plan Agent LLM call failed: {e}")
        plan = f"⚠️ I couldn't generate the study plan due to an error. Please try again.\n\nError: {e}"

    hitl_notice = (
        "\n\n---\n"
        "⚠️ **[HITL Checkpoint]** Please review this plan. "
        "Click **Approve Plan** above or describe changes needed."
    )

    # Set hitl_approved=False explicitly so app.py HITL detector fires
    return {
        "study_plan": plan,
        "hitl_approved": False,
        "analysis": plan,
        "messages": [
            AIMessage(content=f"📅 **Draft Study Plan:**\n\n{plan}{hitl_notice}")
        ],
    }


# ── Evaluate Agent ────────────────────────────────────────────────────────────

def evaluate_agent_node(state: StudyState) -> dict:
    """
    Evaluate Agent:
    Reviews a student's written answer and provides constructive feedback.
    """
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    difficulty = state.get("difficulty", "intermediate")

    context = "\n\n---\n\n".join(docs[:3]) if docs else "No reference material."

    system = f"""You are a fair and constructive academic evaluator.
The student has submitted an answer for evaluation.

Evaluate at {difficulty} level and provide:
1. **Score**: X/10 with justification
2. **Strengths**: What they got right (be specific)
3. **Gaps**: What is missing or incorrect
4. **Correct Answer**: The ideal answer based on the study material
5. **Improvement Tips**: How to write a better answer next time

Be encouraging but honest. Help the student learn from their mistakes."""

    try:
        evaluation = call_llm(system,
                              f"Evaluate this student answer:\n\n{query}\n\nReference material:\n{context}")
    except Exception as e:
        logger.error(f"Evaluate Agent LLM call failed: {e}")
        evaluation = f"⚠️ I couldn't evaluate your answer due to an error. Please try again.\n\nError: {e}"

    return {
        "evaluation_result": evaluation,
        "analysis": evaluation,
        "messages": [AIMessage(content=evaluation)],
    }


# ── Critic ────────────────────────────────────────────────────────────────────

def critic_node(state: StudyState) -> dict:
    """
    Critic Agent:
    Validates the QA agent's output — checks if it actually answers the question
    and is grounded in the study material.
    """
    query = state["query"]
    analysis = state.get("analysis", "")
    docs = "\n\n---\n\n".join(state.get("retrieved_docs", [])) or "None"

    system = """You are a critical reviewer for a student AI assistant.
Check if the response adequately answers the student's question.

Respond with EXACTLY one of:
PASS: <brief reason why the response is sufficient>
FAIL: <specific gap or issue that needs to be addressed>

Nothing else."""

    try:
        verdict = call_llm(system,
                           f"Question: {query}\n\nSource Material:\n{docs}\n\nResponse to review:\n{analysis}")
    except Exception as e:
        logger.warning(f"Critic LLM call failed: {e}")
        # On critic failure, pass by default to avoid infinite loops
        return {
            "critique_passed": True,
            "iteration": state.get("iteration", 0) + 1,
            "messages": [AIMessage(content="✅ **Quality Check:** Passed (critic unavailable)")],
        }

    # Strip and upper-case to avoid false FAILs from minor whitespace/casing
    passed = verdict.strip().upper().startswith("PASS")

    return {
        "critique_passed": passed,
        "iteration": state.get("iteration", 0) + 1,
        "messages": [
            AIMessage(content=f"{'✅' if passed else '⚠️'} **Quality Check:** {verdict}")
        ],
    }


# ── Synthesizer ───────────────────────────────────────────────────────────────

def synthesizer_node(state: StudyState) -> dict:
    """
    Synthesizer Agent:
    Assembles the final polished response for the student.
    """
    query = state["query"]
    task_type = state.get("task_type", "qa")

    # Read task-specific fields first; fall back to analysis
    task_content_map = {
        "quiz":     state.get("quiz_questions", "") or state.get("analysis", ""),
        "plan":     state.get("study_plan", "") or state.get("analysis", ""),
        "explain":  state.get("explanation", "") or state.get("analysis", ""),
        "evaluate": state.get("evaluation_result", "") or state.get("analysis", ""),
    }

    if task_type in task_content_map:
        # For non-QA tasks the agent content IS the final answer (already polished)
        content = state.get("analysis", "") or task_content_map[task_type]
        return {
            "final_answer": content,
            "messages": [AIMessage(content=content)],
        }

    # For QA: do a final polish pass
    analysis = state.get("analysis", "")
    if not analysis:
        return {
            "final_answer": "I wasn't able to generate a response. Please try again.",
            "messages": [AIMessage(content="No answer generated.")],
        }

    system = """You are a helpful study assistant finalising a response for a student.
Polish the analysis into a clear, well-structured final answer.
- Use bullet points or numbered steps where helpful
- Highlight key terms in **bold**
- End with a brief 'Key Takeaway' summary
- Be encouraging"""

    try:
        final = call_llm(system,
                         f"Student Question: {query}\n\nDraft Analysis:\n{analysis}\n\nProvide the final polished answer.")
    except Exception as e:
        logger.warning(f"Synthesizer polish call failed: {e}")
        final = analysis  # Use unpolished analysis as fallback

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_router(state: StudyState) -> str:
    """Route to planner (for retrieval-based tasks) or directly to specialist."""
    task = state.get("task_type", "qa")

    # All tasks benefit from retrieval when documents are available
    if state.get("has_documents"):
        if task in ("qa", "explain", "quiz", "plan", "evaluate"):
            return "planner"

    # No documents — skip retrieval but still route to the correct agent.
    return _task_to_node(task)


def route_after_planner(state: StudyState) -> str:
    """After planning, go to retriever if docs exist, else go directly to task agent."""
    task = state.get("task_type", "qa")
    if state.get("has_documents"):
        return "retriever"
    return _task_to_node(task)


def route_after_retriever(state: StudyState) -> str:
    """After retrieval, dispatch to the correct specialist agent."""
    task = state.get("task_type", "qa")
    return _task_to_node(task)


def route_after_critic(state: StudyState) -> str:
    """Loop back to retriever if critique failed and iterations remain."""
    if (
        not state.get("critique_passed", False)
        and state.get("iteration", 0) < Config.MAX_ITERATIONS
    ):
        return "retriever"
    return "synthesizer"


def _task_to_node(task: str) -> str:
    """Map task type string to graph node name."""
    mapping = {
        "qa":       "qa_agent",
        "explain":  "explain_agent",
        "quiz":     "quiz_agent",
        "plan":     "plan_agent",
        "evaluate": "evaluate_agent",
    }
    return mapping.get(task, "qa_agent")


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(vector_store: VectorStore) -> StateGraph:
    """
    Wire all agents into a compiled LangGraph workflow.
    The vector_store is captured in a closure for the retriever node.
    """

    def _retriever(state):
        return retriever_node(state, vector_store)

    workflow = StateGraph(StudyState)

    # Add all nodes
    workflow.add_node("task_router",    task_router_node)
    workflow.add_node("planner",        planner_node)
    workflow.add_node("retriever",      _retriever)
    workflow.add_node("qa_agent",       qa_agent_node)
    workflow.add_node("explain_agent",  explain_agent_node)
    workflow.add_node("quiz_agent",     quiz_agent_node)
    workflow.add_node("plan_agent",     plan_agent_node)
    workflow.add_node("evaluate_agent", evaluate_agent_node)
    workflow.add_node("critic",         critic_node)
    workflow.add_node("synthesizer",    synthesizer_node)

    # Entry point
    workflow.set_entry_point("task_router")

    # All valid task node destinations must be listed in every
    # conditional_edges map
    all_agent_destinations = {
        "planner":        "planner",
        "qa_agent":       "qa_agent",
        "explain_agent":  "explain_agent",
        "quiz_agent":     "quiz_agent",
        "plan_agent":     "plan_agent",
        "evaluate_agent": "evaluate_agent",
        "synthesizer":    "synthesizer",
    }

    workflow.add_conditional_edges("task_router", route_after_router, all_agent_destinations)

    planner_destinations = {
        "retriever":      "retriever",
        "qa_agent":       "qa_agent",
        "explain_agent":  "explain_agent",
        "quiz_agent":     "quiz_agent",
        "plan_agent":     "plan_agent",
        "evaluate_agent": "evaluate_agent",
    }
    workflow.add_conditional_edges("planner", route_after_planner, planner_destinations)

    retriever_destinations = {
        "qa_agent":       "qa_agent",
        "explain_agent":  "explain_agent",
        "quiz_agent":     "quiz_agent",
        "plan_agent":     "plan_agent",
        "evaluate_agent": "evaluate_agent",
    }
    workflow.add_conditional_edges("retriever", route_after_retriever, retriever_destinations)

    # QA goes through critic loop
    workflow.add_edge("qa_agent", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retriever": "retriever", "synthesizer": "synthesizer"},
    )

    # All other agents go directly to synthesizer
    workflow.add_edge("explain_agent",  "synthesizer")
    workflow.add_edge("quiz_agent",     "synthesizer")
    workflow.add_edge("plan_agent",     "synthesizer")
    workflow.add_edge("evaluate_agent", "synthesizer")

    # Synthesizer always ends
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
