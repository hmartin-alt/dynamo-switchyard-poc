# Switchyard + Dynamo: Intelligent LLM Routing POC

Demonstrates query-complexity routing over two models served via NVIDIA Dynamo on EKS. Simple queries are handled by an efficient diffusion model; complex queries escalate to a capable MoE model. Routing is performed by NeMo Switchyard using a trained prefill router.

---

## Models

| Role | Model |
|------|-------|
| Efficient | `google/diffusiongemma-26B-A4B-it` |
| Capable | `Qwen/Qwen3.6-35B-A3B-FP8` |

---

## Prerequisites

### Infrastructure
- AWS EKS cluster with **one `g7e.12xlarge` node** available in your deployment AZ
- Node group must support scale-to-zero (set `--nodes-min 0`)
- HuggingFace account with access to both models above

### Tooling
```
kubectl          >= 1.28
eksctl           >= 0.180
helm             >= 3.12
python           >= 3.10
git lfs
cargo / rust     (for building Switchyard)
```

### Credentials
```bash
export AWS_REGION=<your-region>
export CLUSTER_NAME=<your-cluster-name>
export NODEGROUP_NAME=<your-nodegroup-name>
export HF_TOKEN=<your-huggingface-token>
```

### Python Environment

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate

# Install the LLM Router toolkit
git clone https://github.com/NVIDIA-AI-Blueprints/llm-router -b v3 ~/llm-router-v3
cd ~/llm-router-v3 && git lfs install && git lfs pull
pip install -e '.[prefill,training,server]'

# Install data collection dependencies
pip install datasets openai
```

Re-activate this venv at the start of any new session:
```bash
source <repo-root>/.venv/bin/activate
```

---

## 1. Install Dynamo 1.4.0

```bash
helm repo add dynamo https://helm.ngc.nvidia.com/nvidia/ai-dynamo
helm repo update

helm install dynamo-platform dynamo/dynamo-platform \
  --namespace dynamo \
  --create-namespace \
  --version 1.4.0

kubectl get pods -n dynamo
```

---

## 2. Scale Up the GPU Node

```bash
eksctl scale nodegroup \
  --cluster $CLUSTER_NAME \
  --name $NODEGROUP_NAME \
  --nodes 1 --nodes-min 1 \
  -r $AWS_REGION

kubectl get nodes -w
```

---

## 3. Pre-download Models

Models are large (~26 GB and ~35 GB). Pre-download them to the node's host path before starting the workers to avoid download timeouts.

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -n dynamo

kubectl apply -f manifests/dynamo/prefetch-job.yaml -n dynamo
kubectl logs -f -n dynamo -l job-name=prefetch-models
```

Wait for `All models downloaded.` before proceeding.

---

## 4. Deploy the Workers

```bash
kubectl apply -f manifests/dynamo/dgd-efficient.yaml -n dynamo
kubectl apply -f manifests/dynamo/dgd-capable.yaml -n dynamo

# The capable worker (Qwen 35B) takes ~5 minutes to initialize
kubectl get pods -n dynamo -w
```

Expected state:

```
efficient-frontend-...            1/1   Running
efficient-vllmdecodeworker-...    1/1   Running
capable-frontend-...              1/1   Running
capable-vllmdecodeworker-...      1/1   Running
```

---

## 5. Port-Forward the Endpoints

```bash
kubectl port-forward -n dynamo svc/efficient-frontend 8000:8000 &
kubectl port-forward -n dynamo svc/capable-frontend   8001:8000 &
```

Verify both endpoints are healthy:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

---

## 6. Build and Start Switchyard

```bash
git clone https://github.com/NVIDIA-NeMo/switchyard ~/switchyard-src
cd ~/switchyard-src
cargo build --release
```

```bash
cd <repo-root>
export DUMMY_KEY=notused
~/switchyard-src/target/release/switchyard-server \
  --config switchyard/routes.toml \
  --host 127.0.0.1 --port 4000
```

Switchyard is now listening on `localhost:4000`.

---

## 7. Train the Prefill Router

The prefill router runs each query through a small encoder model (`Qwen3.5-0.8B`) to extract hidden states, then uses a trained MLP classifier to predict which model will handle the query better. It requires an offline training step using judged samples from your workload.

> **Minimum:** 3k–5k judged samples per model before training. The steps below use the Amazon ESCI dataset as a proxy workload — replace with your own queries and ground-truth labels for production.

### Collect training data

```bash
cd <repo-root>
pip install datasets openai

python3 scripts/esci/01-download-sample.py        # sample 5000 queries
python3 scripts/esci/02-classify-models.py        # run both models, record correctness
python3 scripts/esci/03-derive-routing-labels.py  # derive efficient/capable routing labels
python3 scripts/prefill-router/00-convert-to-csv.py  # split into train/test CSVs
```

> **DiffusionGemma requires the manual scripts.** It rejects the `temperature` parameter that `model-router collect` passes by default. The scripts above handle this with per-model parameter guards. For standard OpenAI-compatible models, see the [LLM Router Blueprint](https://github.com/NVIDIA-AI-Blueprints/llm-router/tree/v3#collect-training-data) for the built-in `model-router collect` command.

Output CSV format (input to `model-router train`):
```
question,model,isCorrect,output_tokens
"wireless mouse",efficient,1,50
"wireless mouse",capable,1,100
```

### Train

```bash
cd ~/llm-router-v3
ROUTER_DEVICE=cpu model-router train \
  --config <repo-root>/configs/prefill-router/shopping-pool.yaml \
  --data <repo-root>/data/prefill-router/train.csv \
  --output-dir <repo-root>/checkpoints/
```

> **Apple Silicon:** `ROUTER_DEVICE=cpu` is required — training crashes on MPS.

### Evaluate

```bash
ROUTER_DEVICE=cpu model-router evaluate \
  --config <repo-root>/configs/prefill-router/shopping-pool.yaml \
  --checkpoint <repo-root>/checkpoints/prefill_router.pt \
  --data <repo-root>/data/prefill-router/test.csv
```

### Serve the router sidecar

```bash
cd <repo-root>
ROUTER_DEVICE=cpu model-router serve-router \
  --config configs/prefill-router/shopping-pool.yaml \
  --port 8079
```

Test it:

```bash
curl -X POST http://localhost:8079/v1/route \
  -H "Content-Type: application/json" \
  -d '{"question": "wireless mouse"}'
```


---

## LLM Classifier (Alternative to Prefill Router)

If you prefer a zero-training-data approach, Switchyard includes a built-in LLM classifier that scores each query at inference time using a prompt-driven evaluation. It requires no offline training but adds an LLM call to every request.

The classifier prompt and routing threshold are configured in `switchyard/routes.toml`:

```toml
[routes.shopping]
type = "llm_classifier"
classifier_target = "strong"   # model used to score the query
strong_target = "strong"
weak_target   = "weak"
base_threshold = 0.8           # p_solve >= 0.8 → route to efficient

prompt = """
You are a query-complexity forecaster for a shopping assistant router...

# Query complexity rules
- SUP-1 [supported, p_solve 0.85–1.00]: Clear product with specific attributes.
- LIM-1 [unsupported, p_solve 0.10–0.40]: Vague requirements or cross-category constraints.

# Output
Return exactly one JSON object. p_solve must be between 0.00 and 1.00.
"""
```

**Tuning the prompt:** Edit the rules and few-shot examples in the `prompt` field to match your domain. Higher `base_threshold` routes more traffic to the efficient model; lower routes more to capable. Run `scripts/esci/04-update-classifier-prompt.py` to automatically inject few-shot examples derived from your ESCI evaluation data.

---

## Scale Down

```bash
eksctl scale nodegroup \
  --cluster $CLUSTER_NAME \
  --name $NODEGROUP_NAME \
  --nodes 0 --nodes-min 0 \
  -r $AWS_REGION
```

## More Information on Switchyard
- [Switchyard Repo](https://github.com/NVIDIA-NeMo/Switchyard)
- [Prefill Routing ](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/docs/routing_algorithms/prefill_routing.md)
- [LLM Classifier](https://github.com/NVIDIA-NeMo/Switchyard/blob/fbabf51c62793ed0f6af042b60e92ce1cfba083b/docs/routing_algorithms/llm_classifier_routing.md)
