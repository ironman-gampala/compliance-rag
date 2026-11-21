"""Chunk windows stay on a token budget and overlap."""

import unittest

from rag.chunking import chunk_text


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class OffsetTokenizer:
    """Splits on spaces and reports character offsets, like a fast Hugging Face tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, verbose=False):
        offsets = []
        cursor = 0
        for word in text.split(" "):
            start = text.find(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        return {"offset_mapping": offsets}


class ChunkTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(chunk_text("   ", FakeTokenizer()), [])

    def test_short_page_is_one_chunk(self):
        self.assertEqual(chunk_text("only three words", FakeTokenizer(), size=10, overlap=2), ["only three words"])

    def test_overlap_window(self):
        words = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_text(words, FakeTokenizer(), size=4, overlap=1)
        self.assertEqual(
            chunks,
            [
                "w0 w1 w2 w3",
                "w3 w4 w5 w6",
                "w6 w7 w8 w9",
            ],
        )

    def test_original_casing_is_kept(self):
        chunks = chunk_text("Hello WORLD of Fees", OffsetTokenizer(), size=2, overlap=0)
        self.assertEqual(chunks, ["Hello WORLD", "of Fees"])

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", FakeTokenizer(), size=2, overlap=2)


if __name__ == "__main__":
    unittest.main()
