#!/usr/bin/env python3
"""
Step 2: Run both models on the ESCI classification task.
For each query+product pair, ask both Gemma (efficient) and Qwen (capable)
to predict the relevance label (Exact/Substitute/Complement/Irrelevant).
Output: data/esci/predictions.jsonl

Requires: port-forwards running on localhost:8000 (efficient) and localhost:8001 (capable)
"""

import json
import time
import re
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).parent.parent.parent
IN_FILE = REPO_ROOT / "data/esci/sample.jsonl"
OUT_FILE = REPO_ROOT / "data/esci/predictions.jsonl"

EFFICIENT = OpenAI(base_url="http://localhost:8000/v1", api_key="notused")
CAPABLE   = OpenAI(base_url="http://localhost:8001/v1", api_key="notused")

SYSTEM_PROMPT = """You are a product relevance classifier for an e-commerce search engine.

Given a search query and a product title, classify the relevance as exactly one of:
- Exact: The product directly satisfies the query intent
- Substitute: The product could work as a substitute but is not exactly what was searched for
- Complement: The product complements the query item but does not satisfy it
- Irrelevant: The product has no meaningful relation to the query

Return only a JSON object with a single field, for example: {"label": "Exact"}
Do not include any other text or explanation."""

DIFFUSION_MODELS = {"efficient"}  # models that don't support temperature
THINKING_MODELS  = {"capable"}    # Qwen3 models with thinking mode — disable it

def classify(client, model_name, query, product_title, bullet_points):
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

        # Parse label from JSON response
        match = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        label = match.group(1) if match else None
        if label not in ("Exact", "Substitute", "Complement", "Irrelevant"):
            label = None
        return label, latency, raw
    except Exception as e:
        return None, 0.0, str(e)

# Resume from where we left off if interrupted
done_ids = set()
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            done_ids.add((r["query"], r["product_title"]))

examples = []
with open(IN_FILE) as f:
    for line in f:
        examples.append(json.loads(line))

remaining = [e for e in examples if (e["query"], e["product_title"]) not in done_ids]
print(f"Total: {len(examples)} | Already done: {len(done_ids)} | Remaining: {len(remaining)}")

with open(OUT_FILE, "a") as out:
    for i, ex in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {ex['query'][:50]!r}...", end=" ", flush=True)

        eff_label, eff_lat, eff_raw = classify(
            EFFICIENT, "efficient", ex["query"], ex["product_title"], ex["product_bullet_point"]
        )
        cap_label, cap_lat, cap_raw = classify(
            CAPABLE, "capable", ex["query"], ex["product_title"], ex["product_bullet_point"]
        )

        gt = ex["ground_truth"]
        record = {
            "query": ex["query"],
            "product_title": ex["product_title"],
            "ground_truth": gt,
            "efficient_prediction": eff_label,
            "capable_prediction": cap_label,
            "efficient_latency_s": eff_lat,
            "capable_latency_s": cap_lat,
            "efficient_correct": eff_label == gt,
            "capable_correct": cap_label == gt,
        }
        out.write(json.dumps(record) + "\n")
        out.flush()

        status = f"gt={gt} eff={eff_label}({'✓' if eff_label==gt else '✗'}) cap={cap_label}({'✓' if cap_label==gt else '✗'})"
        print(status)

print(f"\nDone. Predictions saved to {OUT_FILE}")
