"""
Phase 3 – Sentence-Aware Text Chunking

Splits documents into chunks that NEVER break mid-sentence.
Code blocks are treated as atomic units, and section headings
force chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize          # noqa: E402

from src.config import CHUNK_SIZE, OVERLAP_SENTENCES
from src.loader import Document


# ── Regex patterns ───────────────────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(
    r"```.*?```",          # fenced code blocks produced by loader
    re.DOTALL,
)
_HEADING_RE = re.compile(
    r"^(\d+\.[\d.]*\s+.+)$",   # e.g.  "9.1. A Word About Names"
    re.MULTILINE,
)

_CODE_PLACEHOLDER = "\x00CODE_BLOCK_{}\x00"


@dataclass
class Chunk:
    """A single text chunk with inherited metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _protect_code_blocks(text: str) -> tuple[str, list[str]]:
    """
    Replace fenced code blocks with placeholders so the sentence
    tokenizer doesn't break them.  Returns (modified_text, code_blocks).
    """
    blocks: list[str] = []

    def _replace(m: re.Match) -> str:
        blocks.append(m.group(0))
        return _CODE_PLACEHOLDER.format(len(blocks) - 1)

    protected = _CODE_BLOCK_RE.sub(_replace, text)
    return protected, blocks


def _restore_code_blocks(text: str, blocks: list[str]) -> str:
    """Put the real code blocks back in place of their placeholders."""
    for i, block in enumerate(blocks):
        text = text.replace(_CODE_PLACEHOLDER.format(i), block)
    return text


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split text on section headings.  Returns a list of
    (heading_or_empty, section_body) tuples.
    """
    parts = _HEADING_RE.split(text)
    # parts alternates: [pre-heading text, heading, body, heading, body, ...]
    sections: list[tuple[str, str]] = []

    if parts[0].strip():
        sections.append(("", parts[0].strip()))

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))
        i += 2

    return sections


def _sentence_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[str]:
    """
    Core sentence-aware chunker.
    Accumulates whole sentences up to *chunk_size* characters,
    then starts a new chunk, overlapping the last N sentences.
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # If this single sentence alone exceeds chunk_size, keep it as its own chunk
        if not current and sent_len > chunk_size:
            chunks.append(sentence)
            continue

        if current and current_len + sent_len + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Overlap: carry last N sentences into the next chunk
            current = current[-overlap_sentences:]
            current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

        current.append(sentence)
        current_len += sent_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[Chunk]:
    """
    Turn a list of Documents into a list of Chunks.

    Strategy:
      1. Protect code blocks from the sentence tokenizer.
      2. Split on section headings (force chunk boundary).
      3. Sentence-tokenize each section and group into chunks.
      4. Restore code blocks.
      5. Propagate metadata + chunk_index.
    """
    all_chunks: list[Chunk] = []

    for doc in docs:
        protected, code_blocks = _protect_code_blocks(doc.text)
        sections = _split_by_headings(protected)

        chunk_idx = 0
        for heading, body in sections:
            text_to_chunk = f"{heading}\n\n{body}".strip() if heading else body
            if not text_to_chunk:
                continue

            raw_chunks = _sentence_chunk(text_to_chunk, chunk_size, overlap_sentences)

            for raw in raw_chunks:
                restored = _restore_code_blocks(raw, code_blocks)
                all_chunks.append(Chunk(
                    text=restored,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_idx,
                        "heading": heading,
                    },
                ))
                chunk_idx += 1

    return all_chunks


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint
    from src.loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    rprint(f"[bold green]{len(chunks)} chunks from {len(docs)} documents[/bold green]\n")

    lengths = [len(c.text) for c in chunks]
    rprint(f"  avg length : {sum(lengths) / len(lengths):.0f} chars")
    rprint(f"  min length : {min(lengths)} chars")
    rprint(f"  max length : {max(lengths)} chars\n")

    # Show a sample chunk
    sample = chunks[10]
    rprint(f"[cyan]Sample chunk #{sample.metadata['chunk_index']}[/cyan]")
    rprint(f"  source  : {sample.metadata['source']}")
    rprint(f"  heading : {sample.metadata['heading']}")
    rprint(sample.text[:500], "…")
