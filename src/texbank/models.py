from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ProblemItem:
    identifier: str
    exercise: str
    source_hint: Optional[str] = None
    answer: Optional[str] = None
    solution: Optional[str] = None
    llm_solution: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def needs_llm_solution(self) -> bool:
        return not bool(self.solution) and not bool(self.llm_solution)


__all__ = ["ProblemItem"]
