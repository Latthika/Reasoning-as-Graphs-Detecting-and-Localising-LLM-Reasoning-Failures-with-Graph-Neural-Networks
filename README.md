# Reasoning-as-Graphs: A Graph Neural Network Pipeline for Detecting and Localising LLM Reasoning Failures

> Course project — Machine Learning with Graphs | Spring 2026 | Rice University
> Authors: Latthika Selvamurugan, Abhirami Kathirvel
> Supervisor: Prof. Arlei Lopes da Silva

---

## What this project does

Large language models sometimes write reasoning that looks fluent but ends in a wrong answer. The mistake is usually hidden in a single step. We turn each reasoning trace into a graph (steps as nodes, four edge types between them), train a Graph Neural Network to predict whether the trace is correct, and then run GNNExplainer to point at the step the model relied on most when it predicted "wrong".

The pipeline is the contribution. The classifier itself is competitive but does not beat a strong hand-crafted graph-feature baseline on AUC-ROC.

---

## Headline results (test set: 293 traces, 75 wrong, 218 correct)

| Model | Accuracy | F1 (wrong class) | AUC-ROC | PR-AUC (wrong) |
|---|---|---|---|---|
| Text baseline (logistic regression on full-trace embedding) | 64.5% | 0.660 | 0.639 | 0.42 |
| Graph-feature baseline (8 hand-crafted features + LR) | 70.3% | 0.720 | **0.791** | **0.55** |
| GCN (ours) | 74.4% | 0.635 | 0.678 | 0.46 |
| GAT (ours) | 74.3% | 0.634 | 0.678 | 0.46 |

PR-AUC values are recomputed from the saved probability outputs and rounded.

### Read this honestly

- The base rate for "correct" is 218 / 293 = 74.4%. The GCN's accuracy equals the base rate. At the default 0.5 threshold the GNN behaves close to a majority-class predictor on this split — high accuracy, low F1 on the wrong class.
- The graph-feature baseline (8 hand-crafted statistics passed to logistic regression) **beats the GCN and GAT on AUC-ROC and PR-AUC**. So learned graph representations did not give us a real ranking gain over hand-crafted graph statistics.
- The text baseline (logistic regression on a single 768-dim embedding of the whole trace) is the weakest model. The graph structure clearly carries signal that text alone does not.
- We do not claim the GNN "outperforms" the baselines. We claim graph structure helps and that the explainer pipeline gives a usable interpretability tool on top of any GNN classifier.

### GNNExplainer finding (with caveats)

After base-rate normalisation (dividing failure counts by the number of traces that even contain that step), the most attended-to failure step shifts. Raw counts say Step 2, normalised rate says early steps (1–3) are over-represented relative to availability. Both views are reported in `Stage7_Explain.ipynb`. We removed the earlier, uncorrected "Step 1 = 25.3%" claim — that was raw count divided by `len(wrong_traces)` and is biased by the fact that nearly every trace has a Step 1.

What we keep: the explainer reliably surfaces a small subset of steps per trace, those steps are not uniformly distributed, and inspection of 20 examples (notebook Stage 7) shows the surfaced step is plausibly where the chain diverged in roughly 60–70% of cases. This is qualitative, on a small sample, and we mark it as such in the paper.

---

## Pipeline

```
Stage 1: Generate CoT traces via Groq API on GSM8K problems
    ↓
Stage 2: Parse each trace into individual reasoning steps
    ↓
Stage 3: Embed each step into a 768-dim vector (all-mpnet-base-v2)
    ↓
Stage 4: Build reasoning graphs with 4 edge types
    ↓
Stage 5: Train GCN and GAT classifiers (+ 2 baselines)
    ↓
Stage 6: Evaluate — accuracy, F1, AUC-ROC, PR-AUC, paired t-tests
    ↓
Stage 7: GNNExplainer — surface the step the model attended to
```

### Dataset numbers

- 6,000 GSM8K problems sampled, sent through LLaMA-3.1-8B-Instant via Groq (temperature 0.6).
- 1,948 traces survived filtering (parseable into steps, non-empty answer, deduped).
- Class distribution: 75% correct, 25% wrong. Imbalanced — we use focal loss to handle this.
- Split: 1,363 train / 292 val / 293 test, stratified by label.

We previously called this "balanced". It is not. README and paper are corrected.

---

## Graph construction

Each reasoning trace becomes a graph with 4 edge types:

| Edge type | `edge_attr` | Description |
|---|---|---|
| Sequential | 0 | Step *i* → Step *i+1* (always added) |
| Semantic | 1 | Steps with cosine similarity > 0.75 |
| Value-reuse | 2 | A number from Step *i* reappears in Step *j* |
| Mentions-number | 3 | A number appears in 3+ steps — all linked |

GAT uses `edge_dim=1` so attention can vary per edge type.

---

## Setup

```bash
conda create -n reasoning_graphs python=3.10 -y
conda activate reasoning_graphs
conda install pytorch cpuonly -c pytorch -y
pip install -r requirements.txt
jupyter notebook
```

A free Groq API key is needed only for Stage 1. Paste it in `notebooks/Stage1_Data_Generation.ipynb` Cell 3. Free tier gives 6,000 requests/day, enough to regenerate all traces.

---

## How to run

### Option A — Stage by stage

```
notebooks/Stage1_Data_Generation.ipynb     → data/gsm8k_with_traces.csv
notebooks/Stage2_Parse_Steps.ipynb         → data/gsm8k_steps.csv
notebooks/Stage3_Embed_Steps.ipynb         → data/gsm8k_embeddings.npy
notebooks/Stage4_Build_Graphs.ipynb        → data/graphs.pt
notebooks/Stage5_Train_GNN.ipynb           → data/results.json + checkpoints
notebooks/Stage6_Evaluate.ipynb            → figures + report
notebooks/Stage7_Explain.ipynb             → data/explanations.json
```

### Option B — Single combined notebook

`Reasoning_as_Graphs_Final.ipynb` runs Stages 2–7 end to end. Stage 1 is excluded because it needs a Groq API key and takes about 17 minutes per 1,000 problems.

---

## Repository structure

```
reasoning-as-graphs/
├── README.md
├── requirements.txt
├── Reasoning_as_Graphs_Final.ipynb
├── notebooks/
│   ├── Stage1_Data_Generation.ipynb
│   ├── Stage2_Parse_Steps.ipynb
│   ├── Stage3_Embed_Steps.ipynb
│   ├── Stage4_Build_Graphs.ipynb
│   ├── Stage5_Train_GNN.ipynb
│   ├── Stage6_Evaluate.ipynb
│   └── Stage7_Explain.ipynb
├── data/
│   └── .gitkeep   (generated files excluded — too large for GitHub)
└── paper/
    └── reasoning_as_graphs.pdf  
```

The `data/` folder is empty in the repo. Run Stage 1 to regenerate `gsm8k_with_traces.csv`, then Stages 2–7 to regenerate everything downstream.

---

## Models

**GCN** — two `GCNConv` layers (768 → 128 → 64), global mean pooling, MLP head.
**GAT** — two `GATConv` layers with `edge_dim=1`, global mean pooling, MLP head.

Both trained with focal loss (per-class α = 0.75 on the wrong class, γ = 2.0), Adam (lr = 5e-4), early stopping with patience 30, model selected by **validation AUC-ROC** (not accuracy — accuracy is misleading on this imbalance). Three seeds averaged. Paired t-tests on the three-seed differences are reported in Stage 5.

---

## What changed from the previous draft

- Reframed the headline. The GNN no longer "outperforms" the baselines. It is competitive on accuracy and worse on AUC-ROC than the graph-feature baseline.
- Removed the "balanced 75/25" wording. 75/25 is imbalanced; that is why we use focal loss.
- Stage 5 model selection moved from validation accuracy to validation AUC-ROC.
- Focal loss now uses per-class α (was a single scalar applied to both classes).
- Evaluation reports PR-AUC for the wrong class. This is the right metric for an imbalanced minority-class problem.
- Stage 5 reports paired t-tests on three-seed differences. The GCN-vs-graph-baseline AUC gap is not significant.
- Stage 7 GNNExplainer numbers are now base-rate normalised. The "Step 1 = 25.3%" headline was an artefact of every trace having a Step 1.
- Filename of the paper PDF is corrected.

---

## Acknowledgements

Supervised by Prof. Arlei Lopes da Silva, Rice University.
LLM inference via Groq free tier. Dataset: GSM8K (Cobbe et al.).


