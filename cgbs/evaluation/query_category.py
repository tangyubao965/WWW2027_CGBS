from __future__ import annotations

from typing import Dict, Iterable, List


def is_cover_oriented_query(query: str) -> bool:
    """
    Following the paper analysis, a query is treated as cover-oriented if it
    explicitly contains the word "cover".
    """
    query = str(query).lower()
    return "cover" in query


def assign_query_category(query: str) -> str:
    """Assign a query into cover-oriented or other."""
    if is_cover_oriented_query(query):
        return "cover-oriented"
    return "other"


def split_indices_by_query_category(queries: Iterable[str]) -> Dict[str, List[int]]:
    """Return indices grouped by query category."""
    groups: Dict[str, List[int]] = {
        "cover-oriented": [],
        "other": [],
    }

    for idx, query in enumerate(queries):
        category = assign_query_category(query)
        groups[category].append(idx)

    return groups


def subset_by_indices(items: List, indices: List[int]) -> List:
    """Select items by indices."""
    return [items[i] for i in indices]