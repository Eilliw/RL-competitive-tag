from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random


@dataclass(frozen=True)
class CheckpointRecord:
    role: str
    generation: int
    path: Path


class CheckpointLeague:
    def __init__(self, seed: int | None = None):
        self._records: list[CheckpointRecord] = []
        self._rng = random.Random(seed)

    def add(self, role: str, generation: int, path: str | Path) -> CheckpointRecord:
        record = CheckpointRecord(role=role, generation=generation, path=Path(path))
        self._records.append(record)
        return record

    def list_role(self, role: str) -> list[CheckpointRecord]:
        return [r for r in self._records if r.role == role]

    def sample(self, role: str, strategy: str = "latest") -> CheckpointRecord:
        records = self.list_role(role)
        if not records:
            raise ValueError(f"No checkpoints available for role={role}")
        if strategy == "latest":
            return max(records, key=lambda r: r.generation)
        if strategy == "historical":
            return self._rng.choice(records)
        raise ValueError(f"Unsupported strategy: {strategy}")
