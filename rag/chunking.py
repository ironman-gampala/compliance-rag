"""Split page text into overlapping token windows."""


def chunk_text(text: str, tokenizer, size: int = 500, overlap: int = 80) -> list[str]:
    """Return chunks of about `size` tokens. Adjacent chunks share `overlap` tokens.

    A chunk never mixes two calls, so the caller can keep one page number per chunk.
    """
    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")
    cleaned = text.strip()
    if not cleaned:
        return []

    offsets = _token_offsets(tokenizer, cleaned)
    if offsets is None:
        return _chunk_by_decode(cleaned, tokenizer, size, overlap)
    if not offsets:
        return []

    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        piece = cleaned[char_start:char_end].strip()
        if piece:
            chunks.append(piece)
        if end == len(offsets):
            break
        start = end - overlap
    return chunks


def _token_offsets(tokenizer, text: str):
    """Character spans for each token. None if this tokenizer cannot report them."""
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )
    except TypeError:
        return None
    offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else encoded.get("offset_mapping")
    return list(offsets)


def _chunk_by_decode(text: str, tokenizer, size: int, overlap: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + size, len(token_ids))
        piece = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if end == len(token_ids):
            break
        start = end - overlap
    return chunks
