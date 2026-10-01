#!/usr/bin/env python3
"""
Step 1: Download ESCI and sample 500 English query-product pairs.
Output: data/esci/sample.jsonl
"""

import json
import random
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent.parent
OUT_FILE = REPO_ROOT / "data/esci/sample.jsonl"
TOTAL_SAMPLES = 5000
VALID_LABELS = {"Exact", "Substitute", "Complement", "Irrelevant"}

print("Downloading ESCI dataset from HuggingFace (tasksource/esci)...")
ds = load_dataset("tasksource/esci", split="test")

print(f"Total examples: {len(ds)}")
print(f"Columns: {ds.column_names}")

# Filter to English (US locale) and non-empty titles
us_examples = [
    ex for ex in ds
    if ex.get("product_locale") == "us"
    and ex.get("query", "").strip()
    and ex.get("product_title", "").strip()
    and ex.get("esci_label") in VALID_LABELS
]
print(f"English examples: {len(us_examples)}")

# Stratified sample: ~125 per label
by_label = defaultdict(list)
for ex in us_examples:
    by_label[ex["esci_label"]].append(ex)

per_label = TOTAL_SAMPLES // len(VALID_LABELS)
sampled = []
for label, examples in by_label.items():
    random.seed(42)
    take = min(per_label, len(examples))
    sampled.extend(random.sample(examples, take))

random.shuffle(sampled)
print(f"Sampled {len(sampled)} examples ({per_label} per label)")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    for ex in sampled:
        record = {
            "query": ex["query"].strip(),
            "product_title": ex["product_title"].strip(),
            "product_bullet_point": (ex.get("product_bullet_point") or "").strip()[:300],
            "ground_truth": ex["esci_label"],
            "esci_label": ex["esci_label"],
        }
        f.write(json.dumps(record) + "\n")

print(f"Saved to {OUT_FILE}")
