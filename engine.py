"""
MoME Engine — Mixture of Memory Experts core.
Sparse-gated personal memory with online-learning router.

Requires: numpy, scikit-learn
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)

# ─── Expert Names ──────────────────────────────────────────────────────────

EXPERT_NAMES = ["identity", "knowledge", "projects", "preferences"]

# ─── Tiny Embedder ─────────────────────────────────────────────────────────

class TinyEmbedder:
    """Минимальный bag-of-words эмбеддер без внешних зависимостей."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.word_vectors: Dict[str, np.ndarray] = {}
        self.rng = np.random.RandomState(42)

    def _get_word_vec(self, word: str) -> np.ndarray:
        if word not in self.word_vectors:
            v = self.rng.randn(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            self.word_vectors[word] = v
        return self.word_vectors[word]

    def embed(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = [self._get_word_vec(w) for w in words]
        vec = np.mean(vecs, axis=0).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        return vec


# ─── Memory Expert ─────────────────────────────────────────────────────────

class MemoryExpert:
    """Хранилище памяти эксперта с векторным поиском."""

    def __init__(self, name: str, embedder: TinyEmbedder, store_dir: Path):
        self.name = name
        self.embedder = embedder
        self.path = store_dir / f"{name}.json"
        self.memories: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                    for m in data:
                        m["embedding"] = np.array(m["embedding"], dtype=np.float32)
                    self.memories = data
            except Exception:
                self.memories = []

    def save(self) -> None:
        data = []
        for m in self.memories:
            entry = dict(m)
            entry["embedding"] = m["embedding"].tolist()
            data.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def write(self, text: str) -> None:
        with self._lock:
            embedding = self.embedder.embed(text)
            self.memories.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "embedding": embedding,
                "timestamp": time.time(),
                "access_count": 0,
            })
            self.save()

    def search(self, query: str, top_k: int = 3) -> List[str]:
        with self._lock:
            if not self.memories:
                return []
            q_emb = self.embedder.embed(query)
            scores = [float(np.dot(q_emb, m["embedding"])) for m in self.memories]
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = []
            for i in top_idx:
                self.memories[i]["access_count"] += 1
                results.append(self.memories[i]["text"])
            self.save()
            return results

    def all(self) -> List[str]:
        with self._lock:
            return [m["text"] for m in self.memories]

    def count(self) -> int:
        with self._lock:
            return len(self.memories)

    def delete(self, text_contains: str) -> None:
        with self._lock:
            self.memories = [m for m in self.memories if text_contains not in m["text"]]
            self.save()


# ─── Online Router ─────────────────────────────────────────────────────────

class MemoryRouter:
    """Online-learning роутер экспертов через SGDClassifier."""

    def __init__(self, embedder: TinyEmbedder):
        self.embedder = embedder
        self.classifier = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=0.001,
            learning_rate='adaptive', eta0=0.01,
            warm_start=True, random_state=42,
        )
        self._fitted = False
        self._classes = np.array(EXPERT_NAMES)

    def predict(self, query: str, top_k: int = 2) -> List[tuple[str, float]]:
        emb = self.embedder.embed(query).reshape(1, -1)
        if not self._fitted:
            return [(name, 0.5) for name in EXPERT_NAMES[:top_k]]
        probs = self.classifier.predict_proba(emb)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(EXPERT_NAMES[i], float(probs[i])) for i in top_indices]

    def update(self, query: str, feedback: Dict[str, float]) -> None:
        emb = self.embedder.embed(query).reshape(1, -1)
        best_expert = max(feedback, key=feedback.get)
        target = np.array([best_expert])
        if not self._fitted:
            self.classifier.partial_fit(emb, target, classes=self._classes)
            self._fitted = True
        else:
            self.classifier.partial_fit(emb, target)


# ─── MoME Engine ───────────────────────────────────────────────────────────

class MomeEngine:
    """Ядро MoME: роутер + эксперты + извлечение фактов."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = TinyEmbedder(dim=384)
        self.router = MemoryRouter(self.embedder)
        self.experts = {
            name: MemoryExpert(name, self.embedder, store_dir)
            for name in EXPERT_NAMES
        }

    def query(self, text: str, top_k: int = 2) -> str:
        """Получить релевантный контекст. Возвращает форматированную строку."""
        selected = self.router.predict(text, top_k=top_k)
        parts = []
        for name, confidence in selected:
            results = self.experts[name].search(text, top_k=3)
            if results:
                lines = "\n".join(f"• {r}" for r in results)
                parts.append(f"[{name.upper()}] (conf: {confidence:.2f}):\n{lines}")
        return "\n\n".join(parts) if parts else ""

    def store_fact(self, expert: str, fact: str) -> bool:
        """Сохранить факт в указанного эксперта."""
        if expert in self.experts and fact:
            self.experts[expert].write(fact)
            return True
        return False

    def learn(self, query: str, expert_ratings: Dict[str, float]) -> None:
        """Обучить роутер на feedback."""
        self.router.update(query, expert_ratings)

    def extract_and_store(self, query: str) -> int:
        """Извлечь факты из запроса через regex и сохранить."""
        facts = []

        name_match = re.search(r'(?:меня\s+зовут|мо[её]\s+имя)\s+([^,\.!?]+)', query, re.IGNORECASE)
        if name_match:
            facts.append(("identity", f"Меня зовут {name_match.group(1).strip()}."))

        city_match = re.search(r'(?:я\s+(?:из|живу\s+в)\s+)([^,\.!?]+)', query, re.IGNORECASE)
        if city_match:
            facts.append(("identity", f"Я живу в {city_match.group(1).strip()}."))

        work_match = re.search(r'(?:занимаюсь|работаю\s+над|делаю|пишу)\s+([^,\.!?]+)', query, re.IGNORECASE)
        if work_match:
            facts.append(("projects", f"Работает над {work_match.group(1).strip()}."))

        project_match = re.search(r'(?:проект|мой\s+проект|у\s+меня\s+(?:есть\s+)?проект)\s+([^,\.!?]+)', query, re.IGNORECASE)
        if project_match:
            facts.append(("projects", f"Проект: {project_match.group(1).strip()}."))

        skill_match = re.search(r'(?:знаю|умею|использую|пишу\s+на|стек|технологии?)\s*[\:\-]?\s*([^,\.!?]+)', query, re.IGNORECASE)
        if skill_match:
            facts.append(("knowledge", f"Знает/использует: {skill_match.group(1).strip()}."))

        like_match = re.search(r'(?:нравится|люблю|предпочитаю)\s+([^,\.!?]+)', query, re.IGNORECASE)
        if like_match:
            facts.append(("preferences", f"Предпочитает {like_match.group(1).strip()}."))

        stored = 0
        for expert, fact in facts:
            if self.store_fact(expert, fact):
                stored += 1
                logger.info("  💾 [%s] запомнил: %s", expert, fact[:60])
        return stored

    def get_stats(self) -> Dict[str, int]:
        return {name: expert.count() for name, expert in self.experts.items()}
