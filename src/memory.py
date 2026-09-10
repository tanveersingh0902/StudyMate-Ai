"""
memory.py — Conversation memory + student learning profile management.

Maintains:
  - Rolling window of recent conversation turns
  - Student learning preferences (difficulty, topics, weak areas)
  - Progress tracking (topics studied, quiz scores)
"""

from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .config import Config


class StudentProfile:
    """
    Tracks a student's learning preferences and progress across the session.
    
    This is like a notebook that remembers:
    - What difficulty level the student prefers
    - Which topics they have studied
    - Their quiz scores
    - What areas they find difficult
    """

    def __init__(self):
        self.difficulty: str = Config.DEFAULT_DIFFICULTY   # beginner/intermediate/advanced
        self.topics_studied: List[str] = []
        self.quiz_scores: List[Dict] = []          # [{"topic": ..., "score": ..., "total": ...}]
        self.weak_areas: List[str] = []
        self.preferred_style: str = "balanced"     # visual/analytical/balanced
        self.total_questions_answered: int = 0

    def update_difficulty(self, level: str):
        if level in ("beginner", "intermediate", "advanced"):
            self.difficulty = level

    def record_quiz(self, topic: str, score: int, total: int):
        self.quiz_scores.append({"topic": topic, "score": score, "total": total})
        self.total_questions_answered += total
        # Mark as weak area if score < 60%
        if total > 0 and score / total < 0.6:
            if topic not in self.weak_areas:
                self.weak_areas.append(topic)

    def add_topic_studied(self, topic: str):
        if topic not in self.topics_studied:
            self.topics_studied.append(topic)

    def get_summary(self) -> str:
        """Return a text summary of the student's profile for injection into prompts."""
        lines = [
            f"Difficulty Preference: {self.difficulty}",
            f"Topics Studied: {', '.join(self.topics_studied) if self.topics_studied else 'None yet'}",
            f"Weak Areas: {', '.join(self.weak_areas) if self.weak_areas else 'None identified'}",
            f"Total Questions Answered: {self.total_questions_answered}",
        ]
        if self.quiz_scores:
            recent = self.quiz_scores[-3:]
            lines.append("Recent Quiz Scores: " +
                ", ".join(f"{q['topic']}: {q['score']}/{q['total']}" for q in recent))
        return "\n".join(lines)

    @property
    def average_score_pct(self) -> float:
        if not self.quiz_scores:
            return 0.0
        total_correct = sum(q["score"] for q in self.quiz_scores)
        total_possible = sum(q["total"] for q in self.quiz_scores)
        return (total_correct / total_possible * 100) if total_possible else 0.0


class ConversationMemory:
    """
    Stores conversation turns with a rolling window.

    - Keeps the last `max_turns` complete exchanges.
    - Older turns are compressed into a summary.
    - Exposes get_context() for injection into prompts.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []
        self._summary: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(self, user_message: str, ai_response: str):
        """Record a completed user ↔ AI exchange."""
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "ai", "content": ai_response})

        if len(self._history) > self.max_turns * 2:
            self._compress()

    def get_context(self) -> str:
        """Return formatted conversation history for prompt injection."""
        parts = []
        if self._summary:
            parts.append(f"[Earlier conversation summary]\n{self._summary}\n")

        recent = self._history[-(self.max_turns * 2):]
        for msg in recent:
            prefix = "Student" if msg["role"] == "user" else "StudyMate"
            parts.append(f"{prefix}: {msg['content']}")

        return "\n".join(parts) if parts else "No prior conversation."

    def get_messages(self) -> List[BaseMessage]:
        """Return recent history as LangChain BaseMessage objects."""
        messages: List[BaseMessage] = []
        recent = self._history[-(self.max_turns * 2):]
        for msg in recent:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def clear(self):
        """Reset memory (new study session)."""
        self._history = []
        self._summary = ""

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2

    # ── Private ───────────────────────────────────────────────────────────────

    def _compress(self):
        """Summarise the oldest half of history to save token space."""
        half = len(self._history) // 2
        old_turns = self._history[:half]
        self._history = self._history[half:]

        formatted = "\n".join(
            f"{'Student' if m['role'] == 'user' else 'StudyMate'}: {m['content']}"
            for m in old_turns
        )

        try:
            from .llm_factory import call_llm
            summary_text = call_llm(
                system="You are a helpful assistant that summarises study conversations concisely.",
                user=(
                    "Summarise the following study conversation in 3-4 sentences, "
                    "capturing main topics and conclusions:\n\n" + formatted
                ),
                temperature=0.0,
            )
            self._summary = (self._summary + " " + summary_text).strip() if self._summary else summary_text
        except Exception:
            self._summary = formatted[:500] + "…"
