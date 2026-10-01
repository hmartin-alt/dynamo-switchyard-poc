#!/usr/bin/env python3
"""
Step 3: Derive routing labels from model predictions and split into tuning/eval sets.

Routing label logic:
  efficient_correct and capable_correct  → "efficient"  (simple enough for the cheap model)
  NOT efficient_correct and capable_correct → "capable"  (needs the stronger model)
  Both wrong → excluded

Output:
  data/esci/tuning.jsonl   (~50 examples, balanced efficient/capable)
  data/esci/eval.jsonl     (remaining labeled examples)
  data/esci/summary.json   (label counts and accuracy stats)
"""

import json
import random
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).parent.parent.parent
IN_FILE  = REPO_ROOT / "data/esci/predictions.jsonl"
TUNING   = REPO_ROOT / "data/esci/tuning.jsonl"
EVAL     = REPO_ROOT / "data/esci/eval.jsonl"
SUMMARY  = REPO_ROOT / "data/esci/summary.json"

TUNING_SIZE = 50  # examples reserved for prompt few-shot tuning

examples = []
with open(IN_FILE) as f:
    for line in f:
        examples.append(json.loads(line))

labeled = []
excluded = 0
for ex in examples:
    eff_ok = ex["efficient_correct"]
    cap_ok = ex["capable_correct"]
    if eff_ok:
        ex["routing_label"] = "efficient"
        labeled.append(ex)
    elif cap_ok:
        ex["routing_label"] = "capable"
        labeled.append(ex)
    else:
        excluded += 1

label_counts = Counter(ex["routing_label"] for ex in labeled)
print(f"Total predictions:  {len(examples)}")
print(f"Excluded (both wrong): {excluded}")
print(f"Labeled: {len(labeled)} → efficient={label_counts['efficient']} capable={label_counts['capable']}")

# Break down by ESCI label
print("\nBreakdown by ESCI label:")
print(f"  {'Label':<12} {'Total':>6} {'Eff correct':>12} {'Cap correct':>12} {'Only Eff':>9} {'Only Cap':>9} {'Both wrong':>10}")
for esci_label in ("Exact", "Substitute", "Complement", "Irrelevant"):
    subset = [e for e in examples if e["ground_truth"] == esci_label]
    eff_ok  = sum(1 for e in subset if e["efficient_correct"])
    cap_ok  = sum(1 for e in subset if e["capable_correct"])
    only_eff = sum(1 for e in subset if e["efficient_correct"] and not e["capable_correct"])
    only_cap = sum(1 for e in subset if e["capable_correct"] and not e["efficient_correct"])
    both_wrong = sum(1 for e in subset if not e["efficient_correct"] and not e["capable_correct"])
    n = len(subset)
    print(f"  {esci_label:<12} {n:>6} {eff_ok:>8} ({eff_ok/n:.0%}) {cap_ok:>8} ({cap_ok/n:.0%}) {only_eff:>6} ({only_eff/n:.0%}) {only_cap:>6} ({only_cap/n:.0%}) {both_wrong:>7} ({both_wrong/n:.0%})")

# Stratified split: balance tuning set
random.seed(42)
efficient_pool = [e for e in labeled if e["routing_label"] == "efficient"]
capable_pool   = [e for e in labeled if e["routing_label"] == "capable"]

random.shuffle(efficient_pool)
random.shuffle(capable_pool)

half = TUNING_SIZE // 2
tuning = efficient_pool[:half] + capable_pool[:half]
eval_set = efficient_pool[half:] + capable_pool[half:]

random.shuffle(tuning)
random.shuffle(eval_set)

print(f"\nTuning set:  {len(tuning)} examples ({sum(1 for e in tuning if e['routing_label']=='efficient')} efficient, {sum(1 for e in tuning if e['routing_label']=='capable')} capable)")
print(f"Eval set:    {len(eval_set)} examples")

for path, data in [(TUNING, tuning), (EVAL, eval_set)]:
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")

# Accuracy summary
eff_accuracy = sum(1 for e in examples if e["efficient_correct"]) / len(examples)
cap_accuracy = sum(1 for e in examples if e["capable_correct"]) / len(examples)
avg_eff_latency = sum(e["efficient_latency_s"] for e in examples) / len(examples)
avg_cap_latency = sum(e["capable_latency_s"] for e in examples) / len(examples)

summary = {
    "total": len(examples),
    "excluded_both_wrong": excluded,
    "labeled": len(labeled),
    "routing_label_counts": dict(label_counts),
    "efficient_accuracy": round(eff_accuracy, 3),
    "capable_accuracy": round(cap_accuracy, 3),
    "avg_efficient_latency_s": round(avg_eff_latency, 2),
    "avg_capable_latency_s": round(avg_cap_latency, 2),
    "tuning_size": len(tuning),
    "eval_size": len(eval_set),
}
with open(SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAccuracy  — efficient: {eff_accuracy:.1%}  capable: {cap_accuracy:.1%}")
print(f"Latency   — efficient: {avg_eff_latency:.2f}s  capable: {avg_cap_latency:.2f}s")
print(f"\nSaved tuning → {TUNING}")
print(f"Saved eval   → {EVAL}")
print(f"Saved summary → {SUMMARY}")
