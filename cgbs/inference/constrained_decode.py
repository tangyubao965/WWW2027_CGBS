from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch

from cgbs.identifier.trie import TokenTrie


def build_identifier_trie(
    identifiers: Sequence[str],
    tokenizer,
    add_special_tokens: bool = True,
) -> TokenTrie:
    """Build a token trie from valid identifier strings."""
    return TokenTrie.from_identifier_texts(
        identifiers=identifiers,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )


def make_prefix_allowed_tokens_fn(
    trie: TokenTrie,
    tokenizer,
) -> Callable[[int, torch.Tensor], List[int]]:
    """
    Create a Hugging Face compatible prefix_allowed_tokens_fn.

    The generated prefix may include decoder_start_token_id at the beginning.
    For T5/LongT5, this is usually pad_token_id. The trie is built over the
    actual target identifier tokens, so we remove the decoder start token
    before querying the trie.
    """
    decoder_start_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        del batch_id

        prefix = input_ids.tolist()

        if decoder_start_token_id is not None and len(prefix) > 0:
            if prefix[0] == decoder_start_token_id:
                prefix = prefix[1:]

        if eos_token_id is not None and eos_token_id in prefix:
            return [eos_token_id]

        allowed = trie.get_allowed_tokens(prefix)

        if not allowed:
            if eos_token_id is not None:
                return [eos_token_id]
            return []

        return allowed

    return prefix_allowed_tokens_fn


@torch.no_grad()
def constrained_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_identifiers: Sequence[str],
    num_beams: int = 20,
    num_return_sequences: int = 20,
    max_length: int = 128,
    device: Optional[torch.device] = None,
    **generate_kwargs,
) -> List[List[str]]:
    """
    Generate valid identifiers with trie-constrained decoding.

    Args:
        model:
            A TextOnlyRetriever or compatible model with generate().
        tokenizer:
            Hugging Face tokenizer.
        input_ids:
            Query input ids.
        attention_mask:
            Query attention mask.
        valid_identifiers:
            All valid identifier strings.
        num_beams:
            Beam size.
        num_return_sequences:
            Number of generated candidates per query.
        max_length:
            Maximum generation length.

    Returns:
        A list of candidate identifier lists.
    """
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model = model.to(device)

    trie = build_identifier_trie(
        identifiers=valid_identifiers,
        tokenizer=tokenizer,
        add_special_tokens=True,
    )

    prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
        trie=trie,
        tokenizer=tokenizer,
    )

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        early_stopping=True,
        **generate_kwargs,
    )

    batch_size = input_ids.size(0)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    grouped: List[List[str]] = []
    for i in range(batch_size):
        start = i * num_return_sequences
        end = (i + 1) * num_return_sequences
        candidates = decoded[start:end]

        deduped = []
        seen = set()
        for cand in candidates:
            cand = cand.strip()
            if cand and cand not in seen:
                seen.add(cand)
                deduped.append(cand)

        grouped.append(deduped)

    return grouped