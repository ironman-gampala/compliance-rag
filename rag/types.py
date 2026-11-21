"""Records passed between ingest, retrieval, and the answer step."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    region: str
    doc_type: str
    score: float | None = None

    def label(self) -> str:
        return f"{self.source}, page {self.page}"


@dataclass
class Answer:
    text: str
    hits: list[Chunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)

    def sources(self) -> list[tuple[int, Chunk]]:
        return [(number, self.hits[number - 1]) for number in self.cited]
