"""The answer step refuses to guess when the excerpts do not help."""

import unittest

from rag.answer import NOT_FOUND, answer_question, format_answer
from rag.retrieve import reciprocal_rank_fusion, section_bases_near_topics, supplement_with_specific_sections
from rag.types import Chunk


def chunk(source: str, page: int, text: str) -> Chunk:
    return Chunk(
        id=f"{source}-{page}",
        text=text,
        source=source,
        page=page,
        region="US",
        doc_type="regulation",
    )


class AnswerTests(unittest.TestCase):
    def test_no_chunks_does_not_call_the_model(self):
        def fail(system, user):
            raise AssertionError("the model should not be called")

        answer = answer_question("What is the cap?", [], complete_fn=fail)
        self.assertEqual(answer.text, NOT_FOUND)
        self.assertEqual(answer.cited, [])

    def test_citations_map_to_the_given_chunks(self):
        hits = [chunk("til.pdf", 78, "The fee cannot exceed $25."), chunk("til.pdf", 179, "Safe harbor $35.")]

        def fake(system, user):
            self.assertIn("[1] til.pdf, page 78", user)
            self.assertIn("Not found in the documents", system)
            return "The first late fee is capped at $25 [1], and a repeat fee at $35 [2]."

        answer = answer_question("What is the cap on late fees?", hits, complete_fn=fake)
        self.assertEqual(answer.cited, [1, 2])
        formatted = format_answer(answer)
        self.assertIn("[1] til.pdf, page 78", formatted)
        self.assertIn("[2] til.pdf, page 179", formatted)

    def test_quoted_subsection_is_cited_when_the_model_omits_brackets(self):
        hits = [
            chunk("til.pdf", 141, "Statements go out 21 days before the due date (§1026.5(b)(2)(ii))."),
            chunk("til.pdf", 175, "The cut-off time is 5 p.m. (§1026.10(b)(2)(ii))."),
        ]

        def fake(system, user):
            return "The payment is timely under §1026.5(b)(2)(ii) and §1026.10(b)(2)(ii)."

        answer = answer_question("May the issuer treat it as late?", hits, complete_fn=fake)
        self.assertEqual(answer.cited, [1, 2])

    def test_not_found_sentence_drops_citations(self):
        hits = [chunk("til.pdf", 1, "Unrelated text.")]

        def fake(system, user):
            return "Not found in the documents."

        answer = answer_question("What is the capital of France?", hits, complete_fn=fake)
        self.assertEqual(answer.text, NOT_FOUND)
        self.assertEqual(answer.cited, [])


class SectionKeepTests(unittest.TestCase):
    def test_reranked_list_keeps_a_more_specific_subsection(self):
        narrative = chunk("til.pdf", 78, "Limitations on Penalty Fees—Section 1026.52(b). The fee cannot exceed $25.")
        procedure = chunk(
            "til.pdf",
            179,
            "The issuer does not impose a fee that exceeds the dollar amount associated with the violation (§1026.52(b)(2)(i)(A)).",
        )
        other = chunk("til.pdf", 53, "Payment allocation is covered by section 1026.53.")
        kept = supplement_with_specific_sections([narrative], [narrative, other, procedure])
        self.assertEqual([hit.page for hit in kept], [78, 179])


class TopicSectionTests(unittest.TestCase):
    def test_section_next_to_the_question_topic_is_kept(self):
        text = (
            "Advertising is covered by section 1026.16. "
            + ("Separate mortgage disclosure text. " * 40)
            + "Periodic statements must be mailed at least 21 days before the due date "
            "so a payment is not treated as late (section 1026.5(b))."
        )
        bases = section_bases_near_topics(text, ["day", "late"])
        self.assertIn("1026.5", bases)
        self.assertNotIn("1026.16", bases)


class FusionTests(unittest.TestCase):
    def test_document_high_on_both_lists_wins(self):
        fused = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["b", "c", "a"],
            ]
        )
        self.assertEqual(fused[0], "b")


if __name__ == "__main__":
    unittest.main()
