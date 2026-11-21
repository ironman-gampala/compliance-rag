"""Ask the model to answer from retrieved chunks and nothing else."""

import re

from rag.config import NOT_FOUND
from rag.llm import complete
from rag.retrieve import section_labels
from rag.types import Answer, Chunk

SYSTEM_PROMPT = f"""You are a compliance assistant. Answer using only the numbered excerpts.

Rules:
- Use only facts written in the excerpts. Do not use outside knowledge, and do not replace a dollar figure in an excerpt with a later inflation-adjusted amount.
- If more than one limit in the excerpts applies, apply every limit. The maximum allowed is the lowest limit the excerpts support.
- When an excerpt states a rule and an example, apply that same rule to the amounts in the question. A dollar amount in the question describes the scenario. It is not a legal cap unless an excerpt says it is.
- Cite the excerpt numbers you used, like [1] or [1][2], right after the claim. When an excerpt writes the section number for a rule you use, include that section number in the answer. If another excerpt writes a longer subsection of that same section, quote the longer subsection. Do not invent a section number the excerpts do not write.
- Apply every rule in the excerpts that governs the facts in the question, including when two different sections both apply. The scenario does not have to appear as a worked example. When the question gives a clock time, state the excerpt that sets the cut-off hour on the payment due date and the excerpt that treats a weekend or holiday payment as timely on the next business day, and name both section numbers.
- When two rules both apply and they set different limits, state both and use the stricter one.
- Reply with exactly this sentence and nothing else only when no excerpt states a rule that governs the question:
  {NOT_FOUND}
"""


def build_user_prompt(question: str, hits: list[Chunk]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit.label()} ({hit.region}, {hit.doc_type})\n{hit.text}")
    excerpts = "\n\n".join(blocks)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def parse_citations(text: str, hit_count: int) -> list[int]:
    cited = []
    for raw in re.findall(r"\[(\d+)\]", text):
        number = int(raw)
        if 1 <= number <= hit_count and number not in cited:
            cited.append(number)
    return cited


def _is_not_found(text: str) -> bool:
    body = text.strip()
    if body == NOT_FOUND:
        return True
    # The model sometimes adds quotes or a trailing period. Accept that only
    # when it did not also cite an excerpt.
    compact = body.strip(" \"'").rstrip(".").strip()
    return compact == NOT_FOUND and "[" not in body


def _with_section_citations(text: str, hits: list[Chunk]) -> str:
    """Attach [n] when the model quotes a subsection but forgets the excerpt number."""
    if re.search(r"\[\d+\]", text):
        return text
    lowered = text.lower()
    cited = []
    for number, hit in enumerate(hits, start=1):
        labels = [label for label in section_labels(hit.text) if "(" in label]
        if any(label in lowered for label in labels):
            cited.append(number)
    if not cited:
        return text
    return text.rstrip() + " " + "".join(f"[{number}]" for number in cited)


def answer_question(question: str, hits: list[Chunk], complete_fn=None) -> Answer:
    """Return an answer grounded in `hits`. Empty retrieval never calls the model."""
    if not hits:
        return Answer(text=NOT_FOUND, hits=[], cited=[])
    complete_fn = complete_fn or complete
    raw = _with_section_citations(complete_fn(SYSTEM_PROMPT, build_user_prompt(question, hits)).strip(), hits)
    if _is_not_found(raw):
        return Answer(text=NOT_FOUND, hits=hits, cited=[])
    return Answer(text=raw, hits=hits, cited=parse_citations(raw, len(hits)))


def format_answer(answer: Answer) -> str:
    lines = [answer.text]
    sources = answer.sources()
    if sources:
        lines.append("")
        lines.append("Sources:")
        for number, hit in sources:
            lines.append(f"[{number}] {hit.label()}")
    return "\n".join(lines)
