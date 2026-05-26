"""
MoME — Mixture of Memory Experts plugin for Hermes Agent.

Sparse-gated personal memory with online-learning router.
Replaces monolithic memory context with expert-routed retrieval.

Activate::
    hermes memory setup
    # then select "mome" from the list
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .engine import EXPERT_NAMES, MomeEngine

logger = logging.getLogger(__name__)

# ─── Tool Schemas ─────────────────────────────────────────────────────────

MEMORY_SEARCH_SCHEMA = {
    "name": "mome_search",
    "description": "Search MoME memory experts. Returns relevant facts routed to the right expert.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
            "expert": {
                "type": "string",
                "enum": EXPERT_NAMES + ["all"],
                "description": "Which expert to search (default: auto-routed).",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default: 3).",
            },
        },
        "required": ["query"],
    },
}

MEMORY_STORE_SCHEMA = {
    "name": "mome_store",
    "description": "Store a fact in a specific MoME memory expert.",
    "parameters": {
        "type": "object",
        "properties": {
            "expert": {
                "type": "string",
                "enum": EXPERT_NAMES,
                "description": "Target expert.",
            },
            "fact": {
                "type": "string",
                "description": "The fact to remember.",
            },
        },
        "required": ["expert", "fact"],
    },
}

MEMORY_STATUS_SCHEMA = {
    "name": "mome_status",
    "description": "Show MoME memory status — expert sizes and router learning state.",
    "parameters": {"type": "object", "properties": {}},
}


# ─── MoME Provider ─────────────────────────────────────────────────────────

class MomeProvider(MemoryProvider):
    """MoME — Mixture of Memory Experts for Hermes Agent.

    Sparse-gated personal memory with:
    - 4 experts (identity, knowledge, projects, preferences)
    - Online-learning SGD classifier router
    - Regex-based fact extraction from user queries
    """

    def __init__(self):
        self._engine: Optional[MomeEngine] = None
        self._hermes_home: Optional[Path] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "mome"

    def is_available(self) -> bool:
        return True  # 100% local, no API keys needed

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "store_dir",
                "description": (
                    "Directory for MoME memory storage "
                    "(relative to HERMES_HOME)"
                ),
                "default": "mome_store",
                "required": False,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get(
            "hermes_home",
            os.environ.get("HERMES_HOME", ""),
        )
        if hermes_home:
            self._hermes_home = Path(hermes_home)
        else:
            self._hermes_home = Path.home() / ".hermes"

        store_dir = self._hermes_home / "mome_store"
        self._engine = MomeEngine(store_dir)

        logger.info(
            "MoME initialized: %s (%d experts, %d total facts)",
            store_dir,
            len(self._engine.experts),
            sum(e.count() for e in self._engine.experts.values()),
        )

    def system_prompt_block(self) -> str:
        if not self._engine:
            return ""
        stats = self._engine.get_stats()
        total = sum(stats.values())
        return (
            "## MoME Memory\n"
            f"Active. Experts: "
            f"{', '.join(f'{k}={v}' for k, v in stats.items())} "
            f"total={total}\n"
            "Use `mome_search` to find memories, "
            "`mome_store` to save facts, "
            "`mome_status` for expert info."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._engine:
            return ""
        context = self._engine.query(query)
        if context:
            return f"## MoME Memory Context\n{context}"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._engine:
            return

        def _run() -> None:
            try:
                context = self._engine.query(query)
                with self._prefetch_lock:
                    self._prefetch_result = context
            except Exception as e:
                logger.debug("MoME prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mome-prefetch",
        )
        self._prefetch_thread.start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._engine:
            return
        try:
            self._engine.extract_and_store(user_content)
        except Exception as e:
            logger.debug("MoME sync failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            MEMORY_SEARCH_SCHEMA,
            MEMORY_STORE_SCHEMA,
            MEMORY_STATUS_SCHEMA,
        ]

    def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs,
    ) -> str:
        if not self._engine:
            return json.dumps({"error": "MoME not initialized"})

        if tool_name == "mome_search":
            return self._handle_search(args)
        elif tool_name == "mome_store":
            return self._handle_store(args)
        elif tool_name == "mome_status":
            return self._handle_status()

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        expert_filter = args.get("expert", "all")
        top_k = int(args.get("top_k", 3))

        if expert_filter == "all":
            selected = self._engine.router.predict(query, top_k=2)
            results = []
            for name, conf in selected:
                memories = self._engine.experts[name].search(query, top_k=top_k)
                for m in memories:
                    results.append({
                        "expert": name,
                        "memory": m,
                        "confidence": round(conf, 2),
                    })
            return json.dumps({"results": results, "count": len(results)})

        elif expert_filter in self._engine.experts:
            memories = self._engine.experts[expert_filter].search(
                query, top_k=top_k,
            )
            results = [
                {"expert": expert_filter, "memory": m}
                for m in memories
            ]
            return json.dumps({"results": results, "count": len(results)})

        return json.dumps({"error": f"Unknown expert: {expert_filter}"})

    def _handle_store(self, args: Dict[str, Any]) -> str:
        expert = args.get("expert", "")
        fact = args.get("fact", "")
        if expert in self._engine.experts and fact:
            self._engine.experts[expert].write(fact)
            return json.dumps({
                "result": f"Stored in [{expert}]",
                "fact": fact,
            })
        return json.dumps({
            "error": f"Invalid expert '{expert}' or empty fact",
        })

    def _handle_status(self) -> str:
        stats = self._engine.get_stats()
        router_status = (
            "trained"
            if self._engine.router._fitted
            else "untrained (default routing)"
        )
        return json.dumps({
            "expert_counts": stats,
            "total": sum(stats.values()),
            "router": router_status,
            "experts": EXPERT_NAMES,
        })

    def shutdown(self) -> None:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        self._engine = None
        logger.info("MoME shut down")


def register(ctx) -> None:
    """Register MoME as a Hermes memory provider plugin."""
    ctx.register_memory_provider(MomeProvider())
