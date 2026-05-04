# NLP Project: The Reversal Curse — Replication & Extensions

**Repository:** https://github.com/ashishkg0202/princeton-nlp-reversal-curse
**Original paper:** Berglund et al., *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"* — https://arxiv.org/abs/2309.12288
**Original repo:** https://github.com/lukasberglund/reversal_curse
**Team:** Ashish Gupta, Pragyna Akella, Riyan Charania

This repo contains a replication of Berglund et al.'s Experiments 1, 2, and 3 on modern open and closed models, plus two new ablations (prompting on Exp 2; partial bidirectional augmentation on Exp 1). The paper draft and posters are in [`reports/`](reports/).

---

## Headline findings

1. **Exp 2 (celebrity, Llama-3.3-70B-Instruct)**: the curse is largely **elicitation-bound when knowledge is already in pretraining**. Direct reverse acc is 27.6% on the full 1513-pair set; few-shot CoT with 3 demos lifts it to **50.3%**, almost matching the 52.5% forward ceiling. Prompting alone is enough.
2. **Exp 1 (synthetic, Llama-3.1-8B Tinker LoRA)**: the curse is genuinely **per-fact, not a learnable skill**. Mixing in reverse-direction phrasings for some entities (α = 5/10/25/50%) yields ~95% accuracy on those entities (memorization) but **never moves the held-out entities off the floor** (0–3% across all α). Partial bidirectional augmentation does not generalize.
3. **Closed-API models** (GPT-4o, GPT-5.1, GPT-5.4, GPT-5.4-mini) all show the curse on celebrity reverse — none of them can elicit a parent's child reliably from prompting.

Together these triangulate the curse: cheap to recover when the model already memorized the fact, hard to fix at the weights level when it didn't.

---

## Results

### Experiment 2 — Celebrity reversal (Berglund §4)

Test set: `original_repo/data/celebrity_relations/parent_child_pairs.csv` (1513 pairs). Forward query asks "Who is X's child?", reverse asks "Who is X's parent?" Reverse-only prompting ablations are scored on the same 198-pair held-out slice (3 Hemsworth pairs filtered as demo overlap).

**Closed-model direct baseline** (no prompting tricks):

| Model | Forward | Reverse | n |
|---|---|---|---|
| GPT-4o | 55.6% | 29.7% | 1513 |
| GPT-5.1 | 63.4% | 21.7% | 1513 |
| GPT-5.4 | 78.0% | 25.5% | 200 |
| GPT-5.4-mini | 36.0% | 16.0% | 200 |
| Llama-3.3-70B-Instruct (via Tinker) | 58.7% | 27.6% | 1513 |

**Llama-3.3-70B prompting ablations** (reverse direction only):

| Condition | Reverse acc | n |
|---|---|---|
| Direct (baseline) | 25.8% | 198 |
| Hint ("the parent has a famous child") | 51.0% | 198 |
| Zero-shot CoT ("let's think step by step") | 37.4% | 198 |
| Few-shot CoT, k=1 (Obama only) | 33.8% / 34.5% | 198 / 1510 |
| Few-shot CoT, k=2 (+ Musk) | 44.4% / 42.8% | 198 / 1510 |
| Few-shot CoT, k=3 (+ Hemsworth) | **50.5%** / **50.3%** | 198 / 1510 |
| Few-shot CoT, k=4 (+ Andrea→Taylor Swift) | 50.5% / 50.3% | 198 / 1510 |

Monotone improvement 1 → 3, plateau at 3. The k=4 demo (Swift) adds nothing — 3 reverse-direction demos saturate the elicitation. The lift comes from **reverse-direction worked examples** (Musk, Hemsworth), not the answer-format demo (Obama).

### Experiment 1 — Synthetic fictitious-celebrity reversal (Berglund §3)

Tinker LoRA finetune (rank 32, 20 epochs, seed=42) on the 30-entity synthetic dataset shipped with the original repo. p2d = train Name→Description; d2p = train Description→Name. Each entity has 30 templated phrasings in train, 10 in test.

**α = 0% baselines:**

| Model | Direction | Forward | Reverse |
|---|---|---|---|
| Llama-3.1-8B | d2p | 91.7% | 9.3% |
| Llama-3.1-8B | p2d | 66.3% | 0.7% |
| Llama-3.1-70B | d2p | 89.0% | 10.0% |
| Llama-3.1-70B | p2d | 74.3% | 1.7% |

The classic curse: high forward, near-zero reverse. The d2p reverse > p2d reverse asymmetry is real and stable across model sizes — short-name targets are unrecoverable when only seen as prompts; long-description targets get partial credit through language-model fluency.

**Bidirectional augmentation sweep** (Llama-3.1-8B). For each (base direction, α), pick ⌈α × 30⌉ entities, mix in 30 reverse-direction phrasings of each into training, leave the rest forward-only. Eval reverse acc on injected vs held-out entities separately.

p2d-trained:

| α | injected | forward | reverse_injected | reverse_held_out |
|---|---|---|---|---|
| 0% | 0 | 66.3% | — | 0.7% |
| 5% | 2 | 61.7% | 95.0% | 0.0% |
| 10% | 3 | 61.0% | 100.0% | 0.0% |
| 25% | 8 | 69.3% | 100.0% | 0.0% |
| 50% | 15 | 55.3% | 99.3% | 0.0% |

d2p-trained:

| α | injected | forward | reverse_injected | reverse_held_out |
|---|---|---|---|---|
| 0% | 0 | 91.7% | — | 9.3% |
| 5% | 2 | 89.3% | 40.0% | 2.86% |
| 10% | 3 | 90.7% | 40.0% | 1.11% |
| 25% | 8 | 86.7% | 37.5% | 0.0% |
| 50% | 15 | 90.0% | 42.0% | 2.67% |

`reverse_held_out` never moves off the floor at any α tested. Injected memorization is asymmetric and predictable: ~100% on p2d (short name target), ~40% on d2p (long description target, hard to memorize verbatim). At α=50% on p2d, forward acc takes a real hit (-11pp) as reverse rows displace forward signal.

### Experiment 3 — Instruction-tuning reversal (Berglund §5)

Llama-3.1-8B Tinker LoRA, 20 epochs, seed=42. Same vs reverse instruction templates from `original_repo/data/instructions/`.

| Condition | Realized | Unrealized |
|---|---|---|
| Same direction (Q→A guidance) | 84.0% | 89.0% |
| Reverse direction (A→Q guidance) | 7.4% | 10.0% |

Reverse-direction guidance fails to invert at all — the curse manifests in instruction tuning too.

---

## Repository layout

```
code/
  baselines/
    celebrity_api/           Exp 2 closed-API runner (GPT-4o / 5.1 / 5.4 / 5.4-mini)
    llama_inference/         Exp 2 Llama-3.3-70B inference via Tinker
                             (direct + hint + zero-shot CoT + few-shot CoT k=1..4)
    tinker_experiments/      Exp 1 + Exp 3 Tinker LoRA finetune scaffolding
                             (registry-driven; one ExperimentSpec per experiment)
    llama_experiments/       Earlier QLoRA scaffolding (superseded by tinker_experiments)
  build_bidir_exp1_data.py   One-shot: build α-augmented training sets for Exp 1
  build_baseline_report.py   One-shot: assemble the Exp 2 baseline report docx
  results/
    api_eval/                Exp 2 result CSVs + summary JSONs
    tinker_experiments/      Exp 1 + Exp 3 per-run dirs (results.json + loss_log + train.log)

original_repo/               Vendored clone of lukasberglund/reversal_curse
  data/reverse_experiments/june_version_7921032488/
    {p2d,d2p}_prompts_train.jsonl       Exp 1 forward training data
    {p2d,d2p}_prompts_test.jsonl        Exp 1 forward test
    {p2d,d2p}_reverse_prompts_test.jsonl Exp 1 reverse test
    bidir/                              Generated by build_bidir_exp1_data.py
      {p2d,d2p}_aug{5,10,25,50}/        train.jsonl + split test files + injected_entities.json
  data/celebrity_relations/parent_child_pairs.csv     Exp 2 dataset (1513 pairs)
  data/instructions/copypaste_ug100_rg1000_*/         Exp 3 instruction templates

reports/
  The Reversal Curse Is Alive — But Prompting Can Tame It_V2.pdf   Paper draft (latest)
  poster_V4.html                                                   Poster (latest iteration)
  results_summary_2026-05-03.docx                                  Internal notes
```

---

## Environment

**Python** (Tinker + inference): conda env `TinkerEnv` at `C:\Users\Ashish\anaconda3\envs\TinkerEnv\python.exe`.
**API keys** (set in shell, not committed): `TINKER_API_KEY` for Tinker, `OPENAI_API_KEY` for the closed-API runner.

```bash
conda activate TinkerEnv
export TINKER_API_KEY=tml-...     # or `set` on Windows
export OPENAI_API_KEY=sk-...
```

Run all `python -m` invocations from `Project/code/`.

---

## How to reproduce

### Exp 2 — Closed-API celebrity baselines

```bash
# From Project/code/, OPENAI_API_KEY set
python -m baselines.celebrity_api.evaluate --model gpt-4o --samples 10
python -m baselines.celebrity_api.evaluate --model gpt-5.1 --samples 10
# etc — model id is hardcoded; copy-edit if you want a different one
```

Writes `results/api_eval/<model>_reversal_test_results.csv` and `<model>_summary.json`.

### Exp 2 — Llama-3.3-70B prompting ablations (Tinker)

```bash
# direct baseline (forward+reverse on full 1513)
python -m baselines.llama_inference.evaluate \
  --model meta-llama/Llama-3.3-70B-Instruct --concurrency 10

# reverse-only prompting variants
python -m baselines.llama_inference.evaluate \
  --model meta-llama/Llama-3.3-70B-Instruct --condition hint         --concurrency 10
python -m baselines.llama_inference.evaluate \
  --model meta-llama/Llama-3.3-70B-Instruct --condition zeroshot_cot --concurrency 10

# few-shot CoT k-sweep
for K in 1 2 3 4; do
  python -m baselines.llama_inference.evaluate \
    --model meta-llama/Llama-3.3-70B-Instruct --condition fewshot_cot --k $K --concurrency 10
done
```

Add `--max_pairs 200` for a quick (n=200) slice instead of the full 1513.

### Exp 1 baselines + Exp 3 (Tinker LoRA finetune)

```bash
# Smoke test: 1 training step + 4 eval examples, picks one condition
python -m baselines.tinker_experiments.cli --models llama-3.1-8b --only exp1.d2p --dry_run

# Full Exp 1 + Exp 3 on 8B, all conditions
python -m baselines.tinker_experiments.cli --models llama-3.1-8b

# 70B Exp 1 (slow — use --only to scope)
python -m baselines.tinker_experiments.cli --models llama-3.1-70b --only exp1
```

Conditions in the registry: `exp1.d2p`, `exp1.p2d`, `exp1.{d2p,p2d}_aug{5,10,25,50}`, `exp3.same`, `exp3.reverse`. Each saves `results/tinker_experiments/<run_id>/results.json`. Re-runs skip already-completed runs.

### Exp 1 bidirectional augmentation sweep

```bash
# Build the 8 augmented training sets (α ∈ {5,10,25,50}% × {p2d, d2p}).
# Outputs to original_repo/data/reverse_experiments/.../bidir/.
python build_bidir_exp1_data.py

# Then run any subset of the aug conditions
python -m baselines.tinker_experiments.cli --models llama-3.1-8b --only exp1.p2d_aug25
python -m baselines.tinker_experiments.cli --models llama-3.1-8b --only exp1.d2p_aug50
# ... etc
```

Eval automatically splits reverse acc into `reverse_injected` (entities seen bidirectionally) and `reverse_held_out` (forward-only entities). The `injected_entities.json` in each augmented dir lists which 2/3/8/15 entities got reverse rows.

### Tinker CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--models` | both registered | `llama-3.1-8b`, `llama-3.1-70b` |
| `--only` | all conditions | `exp1` / `exp1.d2p` / `exp1.d2p exp3.same` |
| `--epochs` | 20 | training epochs |
| `--lr` | per-condition (`experiments.py`) | LR override |
| `--lora_rank` | 32 | LoRA rank |
| `--seed` | 42 | training seed |
| `--reeval` | off | re-run eval on already-trained weights |
| `--resume_from` | — | Tinker checkpoint URI to resume one run |
| `--dry_run` | off | 1 training step + 4 eval examples |

---

## Evaluation conventions

- **Tinker LoRA runs (Exp 1, Exp 3)**: greedy decode, `prefix_match` on first min(N, 3) normalized words (case/punct insensitive). Same metric Berglund used.
- **Llama-3.3-70B prompting (Exp 2)**: any-of-N starts-with for `direct`/`hint`; `contains` on the span after `"Answer:"` for CoT conditions (with fallback to last non-empty line if no `"Answer:"` found). 10 samples per query.
- **Closed-API (Exp 2)**: any-of-N starts-with on 10 samples per query.

Forward and reverse accuracy are always reported separately — the **gap is the finding**.

---

## Reports & paper

- **Paper draft (latest)**: `reports/The Reversal Curse Is Alive — But Prompting Can Tame It_V2.pdf`
- **Poster (latest)**: `reports/poster_V4.html`
- Earlier internal summaries: `reports/baseline_summary_2026-04-23.docx`, `reports/results_summary_2026-05-03.docx`

---

## Collaboration

The repo is public — anyone can clone, only listed collaborators can push. Add collaborators at [Settings → Manage access](https://github.com/ashishkg0202/princeton-nlp-reversal-curse/settings/access).
