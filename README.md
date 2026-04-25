# CGBS: Cover-Aware Generative Book Search

This repository contains the implementation of **CGBS**, a cover-aware generative framework for book search. CGBS uses cover images during training to enrich book-side knowledge, while keeping retrieval text-only at inference time.

## Overview

CGBS contains two main components.

1. **Cover-aware book memorizer**  
   The memorizer learns book-side knowledge from multi-granularity book text and cover images. It uses a LongT5-based encoder-decoder backbone, a ViT cover encoder, and hierarchical cover-text alignment to generate hierarchical book identifiers.

2. **Text-only retriever**  
   The retriever takes textual user queries as input and generates book identifiers. It inherits cover-aware knowledge from the memorizer through dual-level distillation and book-text replay. At inference time, it uses only textual user queries.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
torch
transformers
datasets
pillow
tqdm
pyyaml
numpy
scikit-learn
yake
accelerate
```

## Data Format

CGBS uses a unified JSONL format. Each line contains one training or evaluation example:

```json
{
  "book_id": "book_001",
  "query": "I remember a book with a blue cover and a small boat.",
  "book_text": "Title: ...\nAuthor: ...\nContent: ...",
  "identifier": "Title | Author | Publisher | [blue sea boat] | Chapter Title | 12_4_37_8",
  "cover_path": "book_001.jpg",
  "cover_caption": "A blue book cover showing a small boat on the sea.",
  "visual_tag": "blue boat sea",
  "split": "train"
}
```

By default, the processed files are expected at:

```text
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
data/processed/identifiers.txt
```

Cover images are expected under the directory specified by `paths.image_dir` in `configs/base.yaml`.

## Dataset Preprocessing

The repository supports three datasets:

1. **WhatsThatBook**
2. **BookCover**
3. **BookSum-Cover**

Each dataset is first converted into the unified intermediate JSONL format. Then the shared preprocessing steps generate cover captions, visual tags, identifiers, and final train/valid/test files.

### WhatsThatBook

WhatsThatBook provides separate files for books and posts, with predefined train/dev/test splits. The preprocessing script reads:

```text
whatsthatbook_14441_gold_books.jsonl
whatsthatbook_14441_gold_posts_train_11552.split_with_val.jsonl
whatsthatbook_14441_gold_posts_dev_1444.split_with_val.jsonl
whatsthatbook_14441_gold_posts_test_1445.split_with_val.jsonl
```

First extract the cover images:

```bash
mkdir -p data/images/whatsthatbook
tar -xf data/raw/WhatsThatBook/whatsthatbook_gold_book_14441_images.tar   -C data/images/whatsthatbook
```

Then preprocess the dataset:

```bash
python scripts/preprocess_whatsthatbook.py   --data_dir data/raw/WhatsThatBook   --output data/processed/whatsthatbook_unified.jsonl   --image_ext .jpg
```

If the extracted image filenames are not in the format `{book_id}.jpg`, update `infer_cover_path()` in `scripts/preprocess_whatsthatbook.py`.

### BookCover

BookCover is loaded from Hugging Face:

```text
kayteekay/bookimg_dataset
```

The dataset contains one split with two fields:

```text
image
text
```

We split it into:

```text
train: 28000
valid: 1000
test: 3600
```

Run:

```bash
python scripts/preprocess_bookcover.py   --dataset_name kayteekay/bookimg_dataset   --output data/processed/bookcover_unified.jsonl   --image_dir data/images/bookcover   --stats_file outputs/analysis/bookcover_stats.json   --train_size 28000   --valid_size 1000   --test_size 3600
```

The script also reports dataset statistics, including split counts and average book-text length.

### BookSum-Cover

BookSum-Cover is built from BookSum text and augmented with metadata and cover images from Open Library.

This preprocessing is split into two steps.

#### Step 1: Preprocess BookSum text

This step extracts the book id, title, author, chapter title, chapter text, summary, query field, and split. It does not access Open Library.

```bash
python scripts/preprocess_booksum_text.py   --input data/raw/BookSum/booksum.jsonl   --output data/processed/booksum_text_unified.jsonl   --stats_file outputs/analysis/booksum_text_stats.json   --train_ratio 0.8   --valid_ratio 0.1
```

#### Step 2: Augment metadata and covers from Open Library

This step uses `title + author` to search Open Library, obtains metadata and cover ids, downloads cover images, and aligns the downloaded covers back to the original BookSum records.

```bash
python scripts/augment_booksum_covers.py   --input data/processed/booksum_text_unified.jsonl   --output data/processed/booksum_cover_unified.jsonl   --image_dir data/images/booksum_cover   --stats_file outputs/analysis/booksum_cover_stats.json   --sleep_seconds 0.2
```

The output contains fields such as:

```json
{
  "book_id": "...",
  "title": "...",
  "author": "...",
  "publisher": "...",
  "chapter_title": "...",
  "chapter_text": "...",
  "book_text": "...",
  "cover_path": "...",
  "openlibrary_key": "...",
  "openlibrary_cover_id": "...",
  "isbn": "...",
  "split": "train"
}
```

The script also reports Open Library match rate and cover coverage.

## Shared Preprocessing Steps

After converting each dataset into the unified intermediate format, run the shared preprocessing steps below.

### 1. Generate Cover Captions

We use Florence-2-base to generate cover captions:

```bash
python scripts/build_cover_captions.py   --input data/processed/<dataset>_unified.jsonl   --output data/processed/<dataset>_with_captions.jsonl   --image_dir data/images/<dataset>   --device cuda
```

### 2. Extract Visual Tags

We use YAKE to extract keywords from cover captions as visual-semantic tags:

```bash
python scripts/build_visual_tags.py   --input data/processed/<dataset>_with_captions.jsonl   --output data/processed/<dataset>_with_visual_tags.jsonl   --num_keywords 6
```

### 3. Build Hierarchical Identifiers

For book-level identifiers:

```bash
python scripts/build_identifiers.py   --input data/processed/<dataset>_with_visual_tags.jsonl   --output data/processed/<dataset>_with_ids.jsonl   --level book
```

For chapter-level identifiers:

```bash
python scripts/build_identifiers.py   --input data/processed/<dataset>_with_visual_tags.jsonl   --output data/processed/<dataset>_with_ids.jsonl   --level chapter   --text_field chapter_text   --num_cluster_levels 10   --branching_factor 100   --random_suffix_size 100
```

For section-level identifiers:

```bash
python scripts/build_identifiers.py   --input data/processed/<dataset>_with_visual_tags.jsonl   --output data/processed/<dataset>_with_ids.jsonl   --level section   --text_field section_text   --num_cluster_levels 10   --branching_factor 100   --random_suffix_size 100
```

The identifier components are connected with vertical bars. The visual-semantic tag is enclosed in square brackets.

Example:

```text
Title | Author | Publisher | [blue sea boat] | Chapter Title | 12_4_37_8_21_4_0_18_3_9_77
```

### 4. Prepare Final Train, Validation, and Test Files

```bash
python scripts/prepare_jsonl.py   --input data/processed/<dataset>_with_ids.jsonl   --output_dir data/processed/<dataset>   --already_split
```

This creates:

```text
data/processed/<dataset>/train.jsonl
data/processed/<dataset>/valid.jsonl
data/processed/<dataset>/test.jsonl
data/processed/<dataset>/identifiers.txt
```

Update `configs/base.yaml` to point to the corresponding dataset directory.

## Configuration

The main configuration file is:

```text
configs/base.yaml
```

Important settings include:

```yaml
model:
  text_model_name: google/long-t5-tglobal-base
  vision_model_name: google/vit-base-patch16-224
  memorizer_max_input_length: 16384
  retriever_max_input_length: 512
  max_target_length: 128
  use_adapters: true
  num_adapters: 1
  adapter_bottleneck_size: 64
  initialize_retriever_from_memorizer: true
  num_beams: 20
  num_return_sequences: 20

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-5
  num_epochs: 5
  eval_batch_size: 1

loss:
  distill_weight: 1.0
  replay_weight: 1.0
  temperature: 2.0
```

LongT5 with 16k input is memory-intensive. We recommend starting with `batch_size: 1` and using gradient accumulation.

## Training

### 1. Train the Cover-Aware Book Memorizer

```bash
python -m cgbs.training.train_memorizer   --config configs/base.yaml
```

The memorizer checkpoint is saved to:

```text
outputs/memorizer/final/
```

### 2. Train the Text-Only Retriever

```bash
python -m cgbs.training.train_retriever   --config configs/base.yaml   --memorizer_path outputs/memorizer/final
```

The retriever checkpoint is saved to:

```text
outputs/retriever/final/
```

## Inference

Run trie-constrained retrieval on the test set:

```bash
python -m cgbs.inference.retrieve   --config configs/base.yaml   --model_path outputs/retriever/final   --split test   --identifier_file data/processed/<dataset>/identifiers.txt
```

Predictions are saved to:

```text
outputs/predictions/test_predictions.jsonl
```

Each record contains:

```json
{
  "book_id": "book_001",
  "query": "...",
  "target_identifier": "...",
  "predictions": ["...", "..."]
}
```

## Evaluation

The default retrieval metrics are:

```text
Hits@1
Hits@10
MRR@20
```

The retrieval script reports these metrics automatically.

## Analysis

### Query Category Analysis

```bash
python analysis/query_type_analysis.py   --prediction_file outputs/predictions/test_predictions.jsonl
```

Cover-oriented queries are identified using a conservative keyword heuristic: queries containing the word `cover` are labeled as cover-oriented.

### Robustness to Invalid Covers

```bash
python analysis/invalid_cover_robustness.py   --prediction_file outputs/predictions/test_predictions.jsonl   --invalid_cover_file data/processed/invalid_cover_ids.txt
```

### Routing Behavior

Export routing scores:

```bash
python analysis/routing_behavior.py export   --config configs/base.yaml   --split test   --output outputs/analysis/routing_scores.jsonl
```

Analyze routing scores:

```bash
python analysis/routing_behavior.py analyze   --score_file outputs/analysis/routing_scores.jsonl
```

### Efficiency Analysis

```bash
python analysis/efficiency_analysis.py   --config configs/base.yaml   --split test   --identifier_file data/processed/<dataset>/identifiers.txt   --num_warmup_batches 2   --num_eval_batches 50   --output outputs/analysis/efficiency.json
```

### Failure Case Analysis

Summarize failure types:

```bash
python analysis/failure_case_analysis.py summarize   --prediction_file outputs/predictions/test_predictions.jsonl   --k 10
```

Extract failure cases:

```bash
python analysis/failure_case_analysis.py extract   --prediction_file outputs/predictions/test_predictions.jsonl   --output outputs/analysis/failure_cases.jsonl   --k 10   --max_cases 50
```

### Chapter Count Bias Analysis

```bash
python analysis/chapter_count_bias.py   --prediction_file outputs/predictions/test_predictions.jsonl   --metadata_file data/processed/book_metadata.jsonl   --small_threshold 5   --large_threshold 20   --output outputs/analysis/chapter_count_bias.json
```

### Visual Tag Variance Analysis

```bash
python analysis/visual_tag_variance.py   --prediction_files   outputs/predictions/tag_seed1_test_predictions.jsonl   outputs/predictions/tag_seed2_test_predictions.jsonl   outputs/predictions/tag_seed3_test_predictions.jsonl   --output outputs/analysis/visual_tag_variance.json
```

### Hyperparameter Analysis

Prepare a JSONL manifest:

```json
{"run_name": "dw0.5_rw1.0", "prediction_file": "outputs/predictions/dw0.5_rw1.0_test_predictions.jsonl", "distill_weight": "0.5", "replay_weight": "1.0"}
{"run_name": "dw1.0_rw1.0", "prediction_file": "outputs/predictions/dw1.0_rw1.0_test_predictions.jsonl", "distill_weight": "1.0", "replay_weight": "1.0"}
{"run_name": "dw2.0_rw1.0", "prediction_file": "outputs/predictions/dw2.0_rw1.0_test_predictions.jsonl", "distill_weight": "2.0", "replay_weight": "1.0"}
```

Run:

```bash
python analysis/hyperparameter_analysis.py   --manifest outputs/analysis/hyperparameter_manifest.jsonl   --sort_metric MRR@20   --output_csv outputs/analysis/hyperparameter_results.csv   --output_json outputs/analysis/hyperparameter_results.json
```

### Case Study Extraction

Extract cases where CGBS succeeds but a baseline fails:

```bash
python analysis/case_study_extraction.py compare   --cgbs_prediction_file outputs/predictions/cgbs_test_predictions.jsonl   --baseline_prediction_file outputs/predictions/gbs_test_predictions.jsonl   --output outputs/analysis/cgbs_vs_gbs_cover_cases.jsonl   --k 10   --max_cases 50   --query_category cover-oriented
```

Extract successful CGBS cases:

```bash
python analysis/case_study_extraction.py success   --cgbs_prediction_file outputs/predictions/cgbs_test_predictions.jsonl   --output outputs/analysis/cgbs_success_cover_cases.jsonl   --k 10   --max_cases 50   --query_category cover-oriented
```

## Notes

- Cover images are used during memorizer training but not as query-side inputs during retrieval.
- The retriever uses textual user queries only at inference time.
- Identifier generation is constrained by a trie built from all valid book identifiers.
- The default query category heuristic is conservative and may under-detect cover-oriented queries.
- Florence-2-base, LongT5, and 16k-token memorizer inputs may require substantial GPU memory.
- Open Library augmentation should be run with a nonzero sleep interval to avoid overly frequent API requests.

## External Resources

- BookSum: https://github.com/salesforce/booksum
- BookCover: https://huggingface.co/datasets/kayteekay/bookimg_dataset
- Open Library Search API: https://openlibrary.org/dev/docs/api/search
- Open Library Covers API: https://openlibrary.org/dev/docs/api/covers

