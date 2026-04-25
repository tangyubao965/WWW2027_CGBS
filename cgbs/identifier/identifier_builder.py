from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional
from typing import Dict, Optional, Union


def normalize_text(text: Optional[str]) -> str:
    """Normalize a text field used in identifiers."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_visual_tag(tag: Optional[str]) -> str:
    """Normalize a visual-semantic tag and remove outer brackets if any."""
    tag = normalize_text(tag)
    if not tag:
        return ""
    if tag.startswith("[") and tag.endswith("]"):
        tag = tag[1:-1].strip()
    return tag


def join_identifier_parts(*parts: str) -> str:
    """Join non-empty identifier components with vertical bars."""
    clean_parts = [normalize_text(p) for p in parts if normalize_text(p)]
    return " | ".join(clean_parts)


@dataclass
class IdentifierRecord:
    """Structured fields for building hierarchical book identifiers."""

    title: str
    author: str = ""
    publisher: str = ""
    visual_tag: str = ""

    chapter_cluster_id: Optional[Union[int, str]] = None
    section_cluster_id: Optional[Union[int, str]] = None

    level: str = "book"


class BookIdentifierBuilder:
    """
    Build hierarchical identifiers for CGBS.

    The identifier has three possible levels:

    book-id:
        title | author | publisher | [visual tag]

    chapter-id:
        book-id | chapter title | chapter_cluster_x

    section-id:
        chapter-id | section title | section_cluster_x

    The visual-semantic tag is enclosed in square brackets to separate
    it from textual metadata and content-derived components.
    """

    VALID_LEVELS = {"book", "chapter", "section"}

    def build_book_id(
        self,
        title: str,
        author: str = "",
        publisher: str = "",
        visual_tag: str = "",
    ) -> str:
        title = normalize_text(title)
        author = normalize_text(author)
        publisher = normalize_text(publisher)
        visual_tag = normalize_visual_tag(visual_tag)

        if not title:
            raise ValueError("title is required for building a book-level identifier.")

        parts = [title]

        if author:
            parts.append(author)
        if publisher:
            parts.append(publisher)
        if visual_tag:
            parts.append(f"[{visual_tag}]")

        return join_identifier_parts(*parts)

    def build_chapter_id(
            self,
            book_id: str,
            chapter_title: str,
            chapter_cluster_id: Optional[Union[int, str]] = None,
    ) -> str:
        book_id = normalize_text(book_id)
        chapter_title = normalize_text(chapter_title)

        if not book_id:
            raise ValueError("book_id is required for building a chapter-level identifier.")
        if not chapter_title:
            raise ValueError("chapter_title is required for building a chapter-level identifier.")

        parts = [book_id, chapter_title]

        if chapter_cluster_id is not None and str(chapter_cluster_id).strip():
            parts.append(normalize_text(str(chapter_cluster_id)))

        return join_identifier_parts(*parts)

    def build_section_id(
            self,
            chapter_id: str,
            section_title: str,
            section_cluster_id: Optional[Union[int, str]] = None,
    ) -> str:
        chapter_id = normalize_text(chapter_id)
        section_title = normalize_text(section_title)

        if not chapter_id:
            raise ValueError("chapter_id is required for building a section-level identifier.")
        if not section_title:
            raise ValueError("section_title is required for building a section-level identifier.")

        parts = [chapter_id, section_title]

        if section_cluster_id is not None and str(section_cluster_id).strip():
            parts.append(normalize_text(str(section_cluster_id)))

        return join_identifier_parts(*parts)

    def build_from_record(self, record: Dict) -> str:
        """Build an identifier from a dictionary record."""
        level = normalize_text(record.get("level", "book")).lower()
        if level not in self.VALID_LEVELS:
            raise ValueError(f"Invalid identifier level: {level}. Expected one of {self.VALID_LEVELS}.")

        book_id = self.build_book_id(
            title=record.get("title", ""),
            author=record.get("author", ""),
            publisher=record.get("publisher", ""),
            visual_tag=record.get("visual_tag", ""),
        )

        if level == "book":
            return book_id

        chapter_id = self.build_chapter_id(
            book_id=book_id,
            chapter_title=record.get("chapter_title", ""),
            chapter_cluster_id=record.get("chapter_cluster_id", None),
        )

        if level == "chapter":
            return chapter_id

        return self.build_section_id(
            chapter_id=chapter_id,
            section_title=record.get("section_title", ""),
            section_cluster_id=record.get("section_cluster_id", None),
        )

    def build_from_dataclass(self, record: IdentifierRecord) -> str:
        """Build an identifier from an IdentifierRecord dataclass."""
        return self.build_from_record(record.__dict__)