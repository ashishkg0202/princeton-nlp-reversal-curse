"""Experiment registry.

Each ExperimentSpec declares everything that varies between experiments:
data layout, eval splits, per-condition LR, post-processing. Adding a new
experiment is one entry here — no branching elsewhere in the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import paths
from .config import DEFAULT_LR


@dataclass(frozen=True)
class ExperimentSpec:
    name:             str
    conditions:       list[str]
    data_dir:         Callable[[str], Path]      # condition -> dataset directory
    train_file:       Callable[[str], str]       # condition -> filename of train split
    eval_files:       Callable[[str], dict[str, str]]   # condition -> {split_name: filename}
    eval_strip:       str | None = None          # strip expected completion at this char
    run_probe:        bool = True                # print sample generations after training
    lr_per_condition: dict[str, float] = field(default_factory=dict)

    def lr_for(self, condition: str) -> float:
        return self.lr_per_condition.get(condition, DEFAULT_LR)


# ---------------------------------------------------------------------------
# Experiment 1: reversal of fictitious-celebrity facts
#
# Conditions:
#   d2p, p2d                                — α=0% baselines (pure forward)
#   p2d_aug{5,10,25,50}, d2p_aug{5,10,25,50} — bidirectional ablation
#     Augmented dirs (built by code/build_bidir_exp1_data.py) live at
#     EXP1_DATA_DIR/bidir/<base>_aug<alpha>/. Eval is split into forward,
#     reverse_injected (entities with reverse rows in train), reverse_held_out
#     (entities forward-only) so we can separate per-fact memorization from
#     generalization of reversal as a skill.
# ---------------------------------------------------------------------------
_EXP1_AUG_ALPHAS    = [5, 10, 25, 50]
_EXP1_BASELINE_CONDS = ["d2p", "p2d"]
_EXP1_AUG_CONDS      = [f"{base}_aug{a}" for base in _EXP1_BASELINE_CONDS for a in _EXP1_AUG_ALPHAS]


def _exp1_data_dir(cond: str):
    if cond in _EXP1_BASELINE_CONDS:
        return paths.EXP1_DATA_DIR
    return paths.EXP1_DATA_DIR / "bidir" / cond


def _exp1_train_file(cond: str) -> str:
    if cond in _EXP1_BASELINE_CONDS:
        return f"{cond}_prompts_train.jsonl"
    return "train.jsonl"


def _exp1_eval_files(cond: str) -> dict[str, str]:
    if cond in _EXP1_BASELINE_CONDS:
        return {
            "forward": f"{cond}_prompts_test.jsonl",
            "reverse": f"{cond}_reverse_prompts_test.jsonl",
        }
    return {
        "forward":          "forward_test.jsonl",
        "reverse_injected": "reverse_test_injected.jsonl",
        "reverse_held_out": "reverse_test_held_out.jsonl",
    }


# Aug conds inherit the LR of their base direction.
_EXP1_LR = {"d2p": 2e-4, "p2d": 1e-4}
_EXP1_LR.update({c: _EXP1_LR[c.split("_")[0]] for c in _EXP1_AUG_CONDS})


EXP1 = ExperimentSpec(
    name="exp1",
    conditions=_EXP1_BASELINE_CONDS + _EXP1_AUG_CONDS,
    data_dir=_exp1_data_dir,
    train_file=_exp1_train_file,
    eval_files=_exp1_eval_files,
    # p2d completions are ~10 words vs ~5 for d2p -> use a lower LR for stability
    lr_per_condition=_EXP1_LR,
)


# ---------------------------------------------------------------------------
# Experiment 3: reversal of QA instructions
# ---------------------------------------------------------------------------
_EXP3_DIRS = {
    "same":    paths.EXP3_DATA_DIR / "copypaste_ug100_rg1000_same_dir",
    "reverse": paths.EXP3_DATA_DIR / "copypaste_ug100_rg1000_main",
}

EXP3 = ExperimentSpec(
    name="exp3",
    conditions=["same", "reverse"],
    data_dir=lambda cond: _EXP3_DIRS[cond],
    train_file=lambda cond: "guidances.jsonl",
    eval_files=lambda cond: {
        "realized":   "realized_examples.jsonl",
        "unrealized": "unrealized_examples.jsonl",
    },
    eval_strip="\n",        # strip "\n\n<END GUIDANCE TEST>" trailer
    run_probe=False,        # exp3 train data is short instructions, probe noise > signal
)


REGISTRY: dict[str, ExperimentSpec] = {
    EXP1.name: EXP1,
    EXP3.name: EXP3,
}


# ---------------------------------------------------------------------------
# Model registry (separate so experiments don't depend on it)
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "llama-3.1-8b":  "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
}
