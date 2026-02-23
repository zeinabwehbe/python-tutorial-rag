"""
Phase 2 – Document Loading & Cleaning

Parses the mirrored Python tutorial HTML files, strips navigation and
boilerplate, and returns clean documents with metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag

from src.config import TUTORIAL_DIR


# Files that are just index / navigation pages with no tutorial content
_SKIP_FILES = {"index.html", "index-2.html"}


@dataclass
class Document:
    """A single cleaned tutorial page."""
    text: str
    metadata: dict = field(default_factory=dict)


def _extract_title(soup: BeautifulSoup) -> str:
    """Pull the page title from the <title> tag, e.g. '9. Classes'."""
    title_tag = soup.find("title")
    if title_tag:
        # Strip the trailing ' — Python 3.x.x documentation'
        raw = title_tag.get_text()
        return raw.split("—")[0].strip().split("&#")[0].strip()
    return "Unknown"


def _extract_body(soup: BeautifulSoup) -> Tag | None:
    """Return the <div class='body' role='main'> element that holds tutorial content."""
    return soup.find("div", class_="body", role="main")


def _clean_text(body: Tag) -> str:
    """
    Convert the body HTML to clean text:
      - Remove script / style tags
      - Convert code blocks to plain text with >>> markers preserved
      - Collapse excessive whitespace
    """
    # Remove unwanted tags
    for tag in body.find_all(["script", "style"]):
        tag.decompose()

    # Replace <pre> code blocks with their text wrapped in markers
    for pre in body.find_all("pre"):
        code_text = pre.get_text()
        pre.replace_with(f"\n```\n{code_text}\n```\n")

    text = body.get_text(separator="\n")

    # Collapse multiple blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace from the whole document
    return text.strip()


def _extract_section_number(title: str) -> str:
    """'9. Classes' → '9', 'Unknown' → ''."""
    m = re.match(r"^(\d+)\.", title)
    return m.group(1) if m else ""


def load_documents(tutorial_dir: Path = TUTORIAL_DIR) -> list[Document]:
    """
    Load all tutorial HTML pages, returning a list of Document objects
    with clean text and metadata.
    """
    docs: list[Document] = []
    html_files = sorted(tutorial_dir.glob("*.html"))

    if not html_files:
        raise FileNotFoundError(
            f"No HTML files found in {tutorial_dir}. "
            "Check that the py_tutorial folder is in place."
        )

    for fpath in html_files:
        if fpath.name in _SKIP_FILES:
            continue

        html = fpath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        title = _extract_title(soup)
        body = _extract_body(soup)
        if body is None:
            continue

        text = _clean_text(body)
        if not text:
            continue

        docs.append(Document(
            text=text,
            metadata={
                "source": fpath.name,
                "title": title,
                "section": _extract_section_number(title),
            },
        ))

    return docs


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich import print as rprint

    documents = load_documents()
    rprint(f"[bold green]Loaded {len(documents)} documents[/bold green]\n")
    for doc in documents[:3]:
        rprint(f"[cyan]{doc.metadata['title']}[/cyan]  ({doc.metadata['source']})")
        rprint(doc.text[:300], "…\n")
