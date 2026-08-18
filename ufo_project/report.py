from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Report:
    title: str
    sections: list[str] = field(default_factory=list)

    def add(self, heading: str, body: str) -> None:
        self.sections.append(f"## {heading}\n\n{body.strip()}\n")

    def write(self, path: Path) -> None:
        content = [f"# {self.title}\n", *self.sections]
        path.write_text("\n".join(content), encoding="utf-8")
