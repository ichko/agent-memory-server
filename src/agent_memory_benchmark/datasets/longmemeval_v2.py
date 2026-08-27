"""Text-only public adapter for LongMemEval-V2."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agent_memory_benchmark.datasets.base import DatasetAdapter
from agent_memory_benchmark.datasets.models import (
    ContextMessage,
    DatasetExample,
    QAPair,
    Session,
)

logger = logging.getLogger(__name__)

TIERS = ("small", "medium")
DOMAINS = ("web", "enterprise")
CATEGORY_MAP = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}


class HaystackGroup:
    """Questions sharing one exact ordered trajectory haystack."""

    def __init__(
        self,
        adapter: LongMemEvalV2Adapter,
        trajectory_ids: tuple[str, ...],
        questions: list[dict[str, Any]],
    ) -> None:
        self._adapter = adapter
        self.trajectory_ids = trajectory_ids
        self.questions = questions

    @property
    def sessions(self) -> list[Session]:
        return self._adapter.sessions_for(self.trajectory_ids)


class LongMemEvalV2Adapter(DatasetAdapter):
    """Load public V2 files downloaded from ``xiaowu0162/longmemeval-v2``."""

    def __init__(
        self,
        tier: str = "small",
        domain: str = "web",
        *,
        root: Path | str | None = None,
        axtree_part_chars: int = 1800,
        max_axtree_chars: int = 0,
    ) -> None:
        if tier not in TIERS:
            raise ValueError(f"Unknown tier {tier!r}; choose from {TIERS}")
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}; choose from {DOMAINS}")
        env_root = os.environ.get("LME_V2_DATA_ROOT")
        self.root = Path(root or env_root or Path.home() / ".cache/memory-bench/lme-v2")
        self.tier = tier
        self.domain = domain
        self.axtree_part_chars = axtree_part_chars
        self.max_axtree_chars = max_axtree_chars
        self._questions: list[dict[str, Any]] | None = None
        self._haystack: dict[str, list[str]] | None = None
        self._trajectories: dict[str, dict[str, Any]] | None = None
        self._sessions: dict[str, Session] = {}

    @property
    def name(self) -> str:
        return f"LongMemEval-V2 ({self.tier}/{self.domain})"

    def _require_files(self) -> None:
        required = (
            self.root / "questions.jsonl",
            self.root / "trajectories.jsonl",
            self.root / "haystacks" / f"lme_v2_{self.tier}.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing LongMemEval-V2 files: "
                + ", ".join(missing)
                + ". Download xiaowu0162/longmemeval-v2 or set LME_V2_DATA_ROOT."
            )

    @property
    def questions(self) -> list[dict[str, Any]]:
        if self._questions is None:
            self._require_files()
            rows = []
            with (self.root / "questions.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("domain") == self.domain:
                        qtype = str(row.get("question_type", ""))
                        row["category"] = CATEGORY_MAP.get(qtype, qtype)
                        row["is_abstention"] = qtype.endswith("-abs")
                        rows.append(row)
            self._questions = rows
        return self._questions

    @property
    def haystack(self) -> dict[str, list[str]]:
        if self._haystack is None:
            self._require_files()
            path = self.root / "haystacks" / f"lme_v2_{self.tier}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            qids = {str(row["id"]) for row in self.questions}
            self._haystack = {
                str(qid): list(ids) for qid, ids in payload.items() if str(qid) in qids
            }
        return self._haystack

    @property
    def trajectories(self) -> dict[str, dict[str, Any]]:
        if self._trajectories is None:
            needed = {item for ids in self.haystack.values() for item in ids}
            found: dict[str, dict[str, Any]] = {}
            logger.info("Scanning V2 trajectories for %d required rows", len(needed))
            with (self.root / "trajectories.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("id") in needed:
                        found[str(row["id"])] = row
                        if len(found) == len(needed):
                            break
            missing = needed - found.keys()
            if missing:
                raise ValueError(f"Missing {len(missing)} referenced trajectories")
            self._trajectories = found
        return self._trajectories

    def haystack_groups(self) -> list[HaystackGroup]:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for question in self.questions:
            qid = str(question["id"])
            if qid not in self.haystack:
                raise ValueError(f"Question {qid} has no haystack")
            groups.setdefault(tuple(self.haystack[qid]), []).append(question)
        return [
            HaystackGroup(self, ids, questions)
            for ids, questions in sorted(
                groups.items(), key=lambda item: str(item[1][0]["id"])
            )
        ]

    def sessions_for(self, trajectory_ids: tuple[str, ...]) -> list[Session]:
        for trajectory_id in trajectory_ids:
            if trajectory_id not in self._sessions:
                self._sessions[trajectory_id] = self._build_session(
                    self.trajectories[trajectory_id]
                )
        return [self._sessions[item] for item in trajectory_ids]

    def _build_session(self, trajectory: dict[str, Any]) -> Session:
        trajectory_id = str(trajectory["id"])
        goal = str(trajectory.get("goal") or "").strip()
        messages = [
            ContextMessage(
                speaker="user",
                text=(
                    f"Task goal: {goal}\n"
                    f"Start URL: {trajectory.get('start_url') or 'unknown'}"
                ),
            )
        ]
        for state in trajectory.get("states", []):
            if not isinstance(state, dict):
                continue
            step = state.get("step", state.get("state_index"))
            url = state.get("url") or ""
            summary = [f"Step {step} — URL: {url}"]
            if state.get("thought"):
                summary.append(f"Thought: {state['thought']}")
            if state.get("action"):
                summary.append(f"Action: {state['action']}")
            messages.append(ContextMessage("assistant", "\n".join(summary)))
            tree = str(state.get("accessibility_tree") or "")
            if self.max_axtree_chars:
                tree = tree[: self.max_axtree_chars]
            for index, chunk in enumerate(self._split(tree), 1):
                messages.append(
                    ContextMessage(
                        "environment",
                        (
                            f"Observation at step {step}, part {index} "
                            f"— URL: {url}\n{chunk}"
                        ),
                    )
                )
        label = f"Trajectory {trajectory_id} — {' '.join(goal.split())[:120]}"
        return Session(label=label, messages=messages)

    def _split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        limit = self.axtree_part_chars
        if limit <= 0:
            return [text]
        parts: list[str] = []
        while text:
            cut = min(limit, len(text))
            if cut < len(text):
                newline = text.rfind("\n", 0, cut)
                if newline > 0:
                    cut = newline
            parts.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return parts

    def load(self) -> list[DatasetExample]:
        return [
            DatasetExample(
                sessions=group.sessions,
                qa_pairs=[
                    QAPair(
                        question=str(row["question"]),
                        answer=str(row.get("answer", "")),
                        question_id=str(row["id"]),
                        question_type=row.get("question_type"),
                        metadata={
                            "domain": self.domain,
                            "tier": self.tier,
                            "category": row.get("category"),
                            "is_abstention": row.get("is_abstention"),
                            "eval_function": row.get("eval_function"),
                        },
                    )
                    for row in group.questions
                ],
                metadata={
                    "domain": self.domain,
                    "tier": self.tier,
                    "trajectory_ids": list(group.trajectory_ids),
                },
            )
            for group in self.haystack_groups()
        ]
