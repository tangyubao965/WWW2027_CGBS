from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class TokenTrie:
    """
    Token-level trie for constrained decoding.

    The trie stores tokenized identifier sequences. During generation,
    it returns the valid next tokens given the current generated prefix.
    """

    def __init__(self, eos_token_id: Optional[int] = None):
        self.root = TrieNode()
        self.eos_token_id = eos_token_id
        self.num_sequences = 0

    def insert(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            raise ValueError("Cannot insert an empty token sequence into the trie.")

        node = self.root
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]

        if not node.is_end:
            self.num_sequences += 1
        node.is_end = True

    def bulk_insert(self, token_sequences: Iterable[Sequence[int]]) -> None:
        for token_ids in token_sequences:
            self.insert(token_ids)

    def _find_node(self, prefix: Sequence[int]) -> Optional[TrieNode]:
        node = self.root
        for token_id in prefix:
            token_id = int(token_id)
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node

    def contains(self, token_ids: Sequence[int]) -> bool:
        node = self._find_node(token_ids)
        return node is not None and node.is_end

    def get_allowed_tokens(self, prefix: Sequence[int]) -> List[int]:
        """
        Return valid next token ids for a given prefix.

        If the prefix is already a complete identifier and eos_token_id is set,
        EOS is also allowed.
        """
        node = self._find_node(prefix)
        if node is None:
            return []

        allowed = list(node.children.keys())

        if node.is_end and self.eos_token_id is not None:
            allowed.append(self.eos_token_id)

        return allowed

    def get_all_sequences(self) -> List[List[int]]:
        sequences: List[List[int]] = []

        def dfs(node: TrieNode, path: List[int]) -> None:
            if node.is_end:
                sequences.append(path.copy())
            for token_id, child in node.children.items():
                path.append(token_id)
                dfs(child, path)
                path.pop()

        dfs(self.root, [])
        return sequences

    def get_prefix_allowed_tokens_fn(self):
        """
        Return a Hugging Face compatible prefix_allowed_tokens_fn.

        This can be passed to model.generate(..., prefix_allowed_tokens_fn=fn).
        """

        def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
            del batch_id

            if hasattr(input_ids, "tolist"):
                prefix = input_ids.tolist()
            else:
                prefix = list(input_ids)

            return self.get_allowed_tokens(prefix)

        return prefix_allowed_tokens_fn

    @classmethod
    def from_identifier_texts(
        cls,
        identifiers: Sequence[str],
        tokenizer,
        add_special_tokens: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> "TokenTrie":
        """
        Build a trie from raw identifier strings.

        For T5-style models, add_special_tokens=True usually appends EOS.
        """
        if eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

        trie = cls(eos_token_id=eos_token_id)

        seen: Set[str] = set()
        for identifier in identifiers:
            identifier = str(identifier).strip()
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)

            token_ids = tokenizer(
                identifier,
                add_special_tokens=add_special_tokens,
            )["input_ids"]

            trie.insert(token_ids)

        return trie