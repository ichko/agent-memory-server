"""Public adapter for LongMemEval v1."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent_memory_benchmark.datasets.base import DatasetAdapter
from agent_memory_benchmark.datasets.models import (
    ContextMessage,
    DatasetExample,
    QAPair,
    Session,
)

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
SPLITS = {
    "oracle": "longmemeval_oracle",
    "small": "longmemeval_s_cleaned",
    "medium": "longmemeval_m_cleaned",
}
_SUPPLEMENTARY_UTF8_RE = re.compile(rb"[\xf0-\xf4][\x80-\xbf]{3}")


class LongMemEvalAdapter(DatasetAdapter):
    """Load a public LongMemEval split from Hugging Face or a local cache."""

    def __init__(
        self,
        split: str = "oracle",
        *,
        cache_dir: Path | str | None = None,
        timeout: float = 300,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {tuple(SPLITS)}")
        self.split = split
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "memory-bench")
        self.timeout = timeout
        self._rows: list[dict] | None = None
        self._examples: list[DatasetExample] | None = None

    @property
    def name(self) -> str:
        return f"LongMemEval ({self.split})"

    def load_raw(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        filename = f"{SPLITS[self.split]}.json"
        path = self.cache_dir / "longmemeval-v1" / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            response = httpx.get(
                f"{BASE_URL}/{filename}",
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            path.write_bytes(response.content)
        raw = _SUPPLEMENTARY_UTF8_RE.sub(b"", path.read_bytes())
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        self._rows = payload
        return payload

    def load(self) -> list[DatasetExample]:
        if self._examples is not None:
            return self._examples
        examples: list[DatasetExample] = []
        for row in self.load_raw():
            sessions = self._sessions(
                row.get("haystack_sessions", []), row.get("haystack_dates", [])
            )
            qid = row.get("question_id")
            qtype = row.get("question_type")
            examples.append(
                DatasetExample(
                    sessions=sessions,
                    qa_pairs=[
                        QAPair(
                            question=str(row["question"]),
                            answer=str(row["answer"]),
                            question_id=str(qid) if qid is not None else None,
                            question_type=qtype,
                        )
                    ],
                    metadata={
                        "question_id": qid,
                        "question_type": qtype,
                        "question_date": row.get("question_date"),
                        "answer_session_ids": row.get("answer_session_ids"),
                    },
                )
            )
        self._examples = examples
        return examples

    @classmethod
    def _sessions(
        cls, raw_sessions: list[list[dict]], dates: list[str]
    ) -> list[Session]:
        sessions: list[Session] = []
        for index, raw_session in enumerate(raw_sessions):
            label = dates[index] if index < len(dates) else f"Session {index + 1}"
            messages = [
                ContextMessage(speaker=str(turn["role"]), text=str(turn["content"]))
                for turn in raw_session
                if isinstance(turn, dict) and "role" in turn and "content" in turn
            ]
            if messages:
                sessions.append(
                    Session(label=label, messages=messages, date=cls._date(label))
                )
        return sessions

    @staticmethod
    def _date(label: str) -> datetime | None:
        match = re.match(r"(\d{4}/\d{2}/\d{2})\s*(?:\(\w+\))?\s*(\d{2}:\d{2})?", label)
        if not match:
            return None
        try:
            return datetime.strptime(
                f"{match.group(1)} {match.group(2) or '00:00'}",
                "%Y/%m/%d %H:%M",
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
