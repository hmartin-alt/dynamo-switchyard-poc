#!/usr/bin/env python3
"""
Convert ESCI predictions.jsonl to the model-router-toolkit CSV format.

The model-router-toolkit expects:
  question, model, isCorrect, output_tokens

One row per (example, model) pair — so each query appears twice,
once for "efficient" and once for "capable".

Splits into 80% train / 20% test CSVs.
"""

import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
IN_FILE   = REPO_ROOT / "data/esci/predictions.jsonl"
OUT_DIR   = REPO_ROOT / "data/prefill-router"
TRAIN_CSV = OUT_DIR / "train.csv"
TEST_CSV  = OUT_DIR / "test.csv"

TRAIN_RATIO = 0.80
# output_tokens is used for cost estimation only, not for training the MLP.
# Use reasonable approximations for each model.
EFFICIENT_OUTPUT_TOKENS = 50
CAPABLE_OUTPUT_TOKENS   = 100

examples = []
with open(IN_FILE) as f:
    for line in f:
        examples.append(json.loads(line))

random.seed(42)
random.shuffle(examples)
split = int(len(examples) * TRAIN_RATIO)
train_set = examples[:split]
test_set  = examples[split:]

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["question", "model", "isCorrect", "output_tokens"]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for ex in rows:
            w.writerow({
                "question":      ex["query"],
                "model":         "efficient",
                "isCorrect":     int(ex["efficient_correct"]),
                "output_tokens": EFFICIENT_OUTPUT_TOKENS,
            })
            w.writerow({
                "question":      ex["query"],
                "model":         "capable",
                "isCorrect":     int(ex["capable_correct"]),
                "output_tokens": CAPABLE_OUTPUT_TOKENS,
            })

write_csv(TRAIN_CSV, train_set)
write_csv(TEST_CSV,  test_set)

print(f"Total examples: {len(examples)}")
print(f"Train: {len(train_set)} queries → {len(train_set)*2} rows → {TRAIN_CSV}")
print(f"Test:  {len(test_set)} queries → {len(test_set)*2} rows → {TEST_CSV}")

print("\nLabel distribution (train):")
for model_key, col in [("efficient", "efficient_correct"), ("capable", "capable_correct")]:
    correct = sum(1 for e in train_set if e[col])
    n = len(train_set)
    print(f"  {model_key}: {correct}/{n} correct ({correct/n:.1%})")
