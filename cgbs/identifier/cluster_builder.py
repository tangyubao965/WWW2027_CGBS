from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class HierarchicalClusterConfig:
    """
    Configuration for hierarchical clustering based identifier codes.

    num_cluster_levels:
        Number of recursive clustering levels.

    branching_factor:
        Maximum number of clusters at each internal node.

    random_suffix_size:
        Range of the final random suffix id. If set to 100, the suffix is
        sampled from [0, 99].

    min_cluster_size:
        If a node contains fewer than this number of samples, recursive
        clustering stops and the remaining levels are padded with 0.
    """

    num_cluster_levels: int = 10
    branching_factor: int = 100
    random_suffix_size: int = 100
    min_cluster_size: int = 2
    max_features: int = 50000
    random_state: int = 42
    code_width: Optional[int] = None


class HierarchicalClusterCodeBuilder:
    """
    Build numeric hierarchical cluster codes for chapter or section texts.

    The algorithm recursively applies KMeans:
    1. Cluster all texts into at most K groups.
    2. For each group, cluster its members again.
    3. Repeat for num_cluster_levels.
    4. Append a final random numeric suffix to distinguish items that
       remain in the same leaf cluster.

    Each text receives a numeric code such as:
        12_04_37_08_19_22_03_91_06_44_17

    The first num_cluster_levels numbers are cluster ids, and the last
    number is the random suffix.
    """

    def __init__(self, config: Optional[HierarchicalClusterConfig] = None):
        self.config = config or HierarchicalClusterConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._rng = random.Random(self.config.random_state)

    def _format_id(self, value: int) -> str:
        if self.config.code_width is None:
            return str(int(value))
        return str(int(value)).zfill(self.config.code_width)

    def _build_features(self, texts: Sequence[str]):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=self.config.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        return self.vectorizer.fit_transform(texts)

    def _choose_num_clusters(self, num_items: int) -> int:
        if num_items < self.config.min_cluster_size:
            return 1
        return min(self.config.branching_factor, num_items)

    def _pad_remaining_levels(
        self,
        codes: Dict[int, List[int]],
        indices: Sequence[int],
        current_level: int,
    ) -> None:
        remaining = self.config.num_cluster_levels - current_level
        for idx in indices:
            codes[idx].extend([0] * remaining)

    def _recursive_cluster(
        self,
        features,
        indices: List[int],
        level: int,
        codes: Dict[int, List[int]],
    ) -> None:
        if level >= self.config.num_cluster_levels:
            return

        num_items = len(indices)
        num_clusters = self._choose_num_clusters(num_items)

        if num_clusters <= 1:
            for idx in indices:
                codes[idx].append(0)
            self._recursive_cluster(
                features=features,
                indices=indices,
                level=level + 1,
                codes=codes,
            )
            return

        sub_features = features[indices]

        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.config.random_state + level,
            n_init=10,
        )
        labels = kmeans.fit_predict(sub_features)

        label_to_indices: Dict[int, List[int]] = {}
        for original_idx, label in zip(indices, labels):
            label = int(label)
            codes[original_idx].append(label)
            label_to_indices.setdefault(label, []).append(original_idx)

        for child_indices in label_to_indices.values():
            self._recursive_cluster(
                features=features,
                indices=child_indices,
                level=level + 1,
                codes=codes,
            )

    def _assign_random_suffixes(self, codes: Dict[int, List[int]]) -> None:
        """
        Append a final random suffix.

        The suffix is sampled within each final cluster path. We try to avoid
        collisions within the same leaf when possible. If the number of items
        in a leaf exceeds random_suffix_size, suffixes are reused deterministically.
        """
        leaf_to_indices: Dict[tuple, List[int]] = {}
        for idx, code in codes.items():
            leaf_to_indices.setdefault(tuple(code), []).append(idx)

        for leaf, indices in leaf_to_indices.items():
            shuffled_suffixes = list(range(self.config.random_suffix_size))
            self._rng.shuffle(shuffled_suffixes)

            for local_pos, idx in enumerate(indices):
                if local_pos < self.config.random_suffix_size:
                    suffix = shuffled_suffixes[local_pos]
                else:
                    suffix = local_pos % self.config.random_suffix_size
                codes[idx].append(suffix)

    def fit_predict(self, texts: Sequence[str]) -> List[str]:
        """
        Build hierarchical numeric cluster codes.

        Returns:
            A list of numeric code strings with the same order as input texts.
            Example:
                ["12_4_37_8_19_22_3_91_6_44_17", ...]
        """
        if len(texts) == 0:
            return []

        clean_texts = [str(t).strip() if t is not None else "" for t in texts]
        features = self._build_features(clean_texts)

        codes: Dict[int, List[int]] = {i: [] for i in range(len(clean_texts))}
        self._recursive_cluster(
            features=features,
            indices=list(range(len(clean_texts))),
            level=0,
            codes=codes,
        )

        for idx in range(len(clean_texts)):
            if len(codes[idx]) < self.config.num_cluster_levels:
                codes[idx].extend([0] * (self.config.num_cluster_levels - len(codes[idx])))
            elif len(codes[idx]) > self.config.num_cluster_levels:
                codes[idx] = codes[idx][: self.config.num_cluster_levels]

        self._assign_random_suffixes(codes)

        code_strings = []
        for idx in range(len(clean_texts)):
            code = codes[idx]
            code_strings.append("_".join(self._format_id(x) for x in code))

        return code_strings


def build_cluster_codes_for_jsonl(
    input_path: str,
    output_path: str,
    text_field: str,
    output_field: str,
    num_cluster_levels: int = 10,
    branching_factor: int = 100,
    random_suffix_size: int = 100,
    min_cluster_size: int = 2,
    max_features: int = 50000,
    random_state: int = 42,
    code_width: Optional[int] = None,
) -> None:
    """
    Read a JSONL file, build hierarchical numeric cluster codes from a text
    field, and write the enriched records to another JSONL file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_id} in {input_path}") from exc

    texts = [record.get(text_field, "") for record in records]

    builder = HierarchicalClusterCodeBuilder(
        HierarchicalClusterConfig(
            num_cluster_levels=num_cluster_levels,
            branching_factor=branching_factor,
            random_suffix_size=random_suffix_size,
            min_cluster_size=min_cluster_size,
            max_features=max_features,
            random_state=random_state,
            code_width=code_width,
        )
    )
    cluster_codes = builder.fit_predict(texts)

    with output_path.open("w", encoding="utf-8") as f:
        for record, cluster_code in zip(records, cluster_codes):
            record[output_field] = cluster_code
            f.write(json.dumps(record, ensure_ascii=False) + "\n")