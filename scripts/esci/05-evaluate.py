What it doesn't fix. When Switchyard sends a question to Qwen, Qwen will "think" first, which your direct Qwen calls didn't. That makes the comparison slightly unfair, and I haven't tested a fix for it.#!/usr/bin/env python3
"""
Step 5: Evaluate routing accuracy comparing three configurations on the held-out eval set.

Configurations:
  1. Gemma-only (efficient):  all queries go to localhost:8000
  2. Qwen-only (capable):     all queries go to localhost:8001
  3. Switchyard-routed:       queries go to localhost:4000 (uses classifier to route)

Metric: classification accuracy on ESCI ground truth labels.
Also reports latency and which queries Switchyard routed to each model.

Requires: port-forwards on :8000 and :8001, Switchyard running on :4000
"""

import json
import time
import re
from pathlib import Path
from collections import Counter, defaultdict

from openai import OpenAI

REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_FILE  = REPO_ROOT / "data/esci/eval.jsonl"
RESULTS    = REPO_ROOT / "data/esci/eval-results.jsonl"
REPORT     = REPO_ROOT / "data/esci/eval-report.json"

EFFICIENT   = OpenAI(base_url="http://localhost:8000/v1", api_key="notused")
CAPABLE     = OpenAI(base_url="http://localhost:8001/v1", api_key="notused")
SWITCHYARD  = OpenAI(base_url="http://localhost:4000/v1", api_key="notused")

SYSTEM_PROMPT = """You are a product relevance classifier for an e-commerce search engine.

Given a search query and a product title, classify the relevance as exactly one of:
- Exact: The product directly satisfies the query intent
- Substitute: The product could work as a substitute but is not exactly what was searched for
- Complement: The product complements the query item but does not satisfy it
- Irrelevant: The product has no meaningful relation to the query

Return only a JSON object with a single field, for example: {"label": "Exact"}
Do not include any other text or explanation."""

DIFFUSION_MODELS = {"efficient", "shopping"}   # models that don't support temperature
THINKING_MODELS  = {"capable"}     # Qwen3 models — disable thinking for clean JSON output

def call_model(client, model_name, query, product_title, bullet_points=""):
    user_msg = f"Query: {query}\nProduct: {product_title}"
    if bullet_points:
        user_msg += f"\nDetails: {bullet_points[:200]}"

    kwargs = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=100,
    )
    if model_name not in DIFFUSION_MODELS:
        kwargs["temperature"] = 0.0
    if model_name in THINKING_MODELS:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        latency = round(time.time() - t0, 2)
        raw = resp.choices[0].message.content.strip()
        routed_model = getattr(resp, "model", model_name)
        match = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        label = match.group(1) if match else None
        if label not in ("Exact", "Substitute", "Complement", "Irrelevant"):
            label = None
        return label, latency, routed_model
    except Exception as e:
        return None, 0.0, str(e)

eval_examples = []
with open(EVAL_FILE) as f:
    for line in f:
        eval_examples.append(json.loads(line))

print(f"Evaluating {len(eval_examples)} examples across 3 configurations...\n")

results = []
for i, ex in enumerate(eval_examples):
    print(f"[{i+1}/{len(eval_examples)}] {ex['query'][:50]!r}", end=" ", flush=True)

    q, pt, bp, gt = ex["query"], ex["product_title"], ex.get("product_bullet_point", ""), ex["ground_truth"]

    eff_label, eff_lat, _   = call_model(EFFICIENT,  "efficient",      q, pt, bp)
    cap_label, cap_lat, _   = call_model(CAPABLE,    "capable",        q, pt, bp)
    sw_label,  sw_lat,  sw_model = call_model(SWITCHYARD, "shopping", q, pt, bp)

    record = {
        "query": q,
        "product_title": pt,
        "ground_truth": gt,
        "routing_label": ex.get("routing_label"),
        "efficient_prediction": eff_label,
        "capable_prediction":   cap_label,
        "switchyard_prediction": sw_label,
        "switchyard_routed_to": sw_model,
        "efficient_latency_s": eff_lat,
        "capable_latency_s":   cap_lat,
        "switchyard_latency_s": sw_lat,
        "efficient_correct":   eff_label == gt,
        "capable_correct":     cap_label == gt,
        "switchyard_correct":  sw_label == gt,
    }
    results.append(record)
    print(f"gt={gt} sw={sw_label}({'✓' if sw_label==gt else '✗'}) routed→{sw_model}")

with open(RESULTS, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Compute summary stats
n = len(results)
eff_acc = sum(1 for r in results if r["efficient_correct"]) / n
cap_acc = sum(1 for r in results if r["capable_correct"]) / n
sw_acc  = sum(1 for r in results if r["switchyard_correct"]) / n

sw_routing = Counter(r["switchyard_routed_to"] for r in results)
avg_eff_lat = sum(r["efficient_latency_s"] for r in results) / n
avg_cap_lat = sum(r["capable_latency_s"] for r in results) / n
avg_sw_lat  = sum(r["switchyard_latency_s"] for r in results) / n

# Routing accuracy: did Switchyard send to the right model?
routing_correct = sum(
    1 for r in results
    if r.get("routing_label") and r["routing_label"] in (r["switchyard_routed_to"] or "")
)

report = {
    "n_examples": n,
    "accuracy": {
        "efficient_only": round(eff_acc, 3),
        "capable_only":   round(cap_acc, 3),
        "switchyard_routed": round(sw_acc, 3),
    },
    "avg_latency_s": {
        "efficient_only": round(avg_eff_lat, 2),
        "capable_only":   round(avg_cap_lat, 2),
        "switchyard_routed": round(avg_sw_lat, 2),
    },
    "switchyard_routing_distribution": dict(sw_routing),
    "routing_label_match_rate": round(routing_correct / n, 3) if n else 0,
}

with open(REPORT, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*50}")
print(f"ACCURACY")
print(f"  Efficient-only:    {eff_acc:.1%}")
print(f"  Capable-only:      {cap_acc:.1%}")
print(f"  Switchyard-routed: {sw_acc:.1%}")
print(f"\nAVG LATENCY")
print(f"  Efficient-only:    {avg_eff_lat:.2f}s")
print(f"  Capable-only:      {avg_cap_lat:.2f}s")
print(f"  Switchyard-routed: {avg_sw_lat:.2f}s")
print(f"\nSWITCHYARD ROUTING DISTRIBUTION")
for model, count in sw_routing.most_common():
    print(f"  {model}: {count} ({count/n:.0%})")
print(f"\nFull report saved to {REPORT}")
