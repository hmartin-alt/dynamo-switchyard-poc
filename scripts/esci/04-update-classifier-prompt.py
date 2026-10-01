#!/usr/bin/env python3
"""
Step 4: Update the Switchyard routes.toml classifier prompt with few-shot examples
derived from the tuning set.

Takes data/esci/tuning.jsonl (50 examples with routing_label=efficient/capable)
and inserts them into the prompt= field in switchyard/routes.toml as concrete
examples that show the classifier what "simple" vs "complex" looks like.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
TUNING_FILE = REPO_ROOT / "data/esci/tuning.jsonl"
TOML_FILE   = REPO_ROOT / "switchyard/routes.toml"

tuning = []
with open(TUNING_FILE) as f:
    for line in f:
        tuning.append(json.loads(line))

# Pick 10 representative few-shot examples (5 efficient, 5 capable)
efficient_ex = [e for e in tuning if e["routing_label"] == "efficient"][:5]
capable_ex   = [e for e in tuning if e["routing_label"] == "capable"][:5]

def format_example(ex):
    label = ex["routing_label"]
    p_solve = "0.90" if label == "efficient" else "0.15"
    rule = "SUP-1" if label == "efficient" else "LIM-1"
    boundary = "supported" if label == "efficient" else "unsupported"
    query = ex["query"].replace('"', '\\"')
    return (
        f'Query: "{query}"\n'
        f'→ {{"crux": "query is {label}", "primary_rule": "{rule}", '
        f'"capability_boundary": "{boundary}", "p_solve": {p_solve}}}'
    )

few_shot_block = "\n\n".join(
    format_example(e) for e in efficient_ex + capable_ex
)

new_prompt = f'''You are a query-complexity forecaster for a shopping assistant router. You receive a shopper\'s query.

Forecast one binary event:

SUCCESS means the efficient (simpler) model handles the shopping query well — the shopper receives a direct, useful product recommendation without needing nuanced reasoning, trade-off analysis, or interpretation of ambiguous requirements. FAILURE means the query requires more sophisticated reasoning and should go to the capable model.

Use only evidence visible in the query itself.

# Assessment procedure

1. State the crux: the key factor that determines whether this query is simple or complex.
2. Select the one rule below that best describes the query complexity. Rule ids are opaque labels.
3. Estimate p_solve: the probability the efficient model handles this query well.

# Query complexity rules

- SUP-1 [supported, p_solve 0.85–1.00]: The query names a specific, well-known product or product category with clear attributes. No interpretation needed. Examples: "wireless mouse", "USB-C charger 65W", "women\'s black running shoes size 8".
- SUP-2 [supported, p_solve 0.75–0.90]: The query has one or two simple, explicit constraints on a clear product category. Examples: "laptop bag under $50", "blue water bottle 32oz BPA-free".
- UNC-1 [uncertain, p_solve 0.40–0.75]: The query has some ambiguity or mild trade-off but a reasonably clear intent. Examples: "comfortable office chair", "headphones for working from home".
- LIM-1 [unsupported, p_solve 0.10–0.40]: The query requires interpreting vague requirements, resolving implicit trade-offs, or navigating cross-category constraints. Examples: "gift for 8-year-old who likes space but no screens under $30", "shoes for standing all day but not running shoes".
- LIM-2 [unsupported, p_solve 0.00–0.15]: The query is highly ambiguous, has many conflicting constraints, or requires deep contextual knowledge. Examples: "something for my mom\'s birthday she\'s into wellness but nothing too woo-woo".

# Examples

{few_shot_block}

# Output

Return exactly one JSON object matching the response schema supplied with the request. Do not include markdown or commentary.

p_solve must be between 0.00 and 1.00. Set capability_boundary to "supported" for SUP rules, "uncertain" for UNC rules, "unsupported" for LIM rules, "unmatched" if no rule fits.'''

toml_content = TOML_FILE.read_text()

# Replace the existing prompt = """...""" block
new_toml = re.sub(
    r'prompt\s*=\s*""".*?"""',
    f'prompt = """\n{new_prompt}\n"""',
    toml_content,
    flags=re.DOTALL,
)

if new_toml == toml_content:
    print("WARNING: Could not find existing prompt block to replace. Check routes.toml format.")
else:
    TOML_FILE.write_text(new_toml)
    print(f"Updated {TOML_FILE} with {len(efficient_ex)} efficient + {len(capable_ex)} capable few-shot examples.")
    print("\nFew-shot examples added:")
    for ex in efficient_ex + capable_ex:
        print(f"  [{ex['routing_label']:9s}] {ex['query'][:60]!r}")
    print("\nRestart Switchyard to apply the updated prompt.")
