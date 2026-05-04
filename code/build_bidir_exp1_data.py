"""Build bidirectional-augmented training sets for Berglund Exp 1.

For each base direction (p2d / d2p), generate augmented training files where a
fraction alpha of the 30 entities have their REVERSE-direction phrasings mixed
in. The remaining entities stay forward-only. This lets us measure separately:
  - reverse acc on the *injected* entities (per-fact memorization)
  - reverse acc on the *held-out* entities (generalization of reversal as a skill)

Output layout (per (base_dir, alpha)):
  data/exp1_bidir/<base>_aug<alpha>/
    train.jsonl                 # original forward + injected reverse rows
    forward_test.jsonl          # copied from original
    reverse_test_injected.jsonl # subset of <base>_reverse_prompts_test on injected entities
    reverse_test_held_out.jsonl # subset on held-out entities
    injected_entities.json      # the K entity names that got reverse rows

Usage (from Project/code/):
  python build_bidir_exp1_data.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_DIR      = PROJECT_ROOT / "original_repo" / "data" / "reverse_experiments" / "june_version_7921032488"
TPL_DIR      = PROJECT_ROOT / "original_repo" / "data" / "reverse_experiments" / "templates"
OUT_ROOT     = SRC_DIR / "bidir"

ALPHAS       = [5, 10, 25, 50]
N_REV_PER_ENTITY = 30
SEED         = 42


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_templates(path: Path) -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and "<name>" in l and "<description>" in l]


def extract_entities(train_path: Path, base: str) -> dict[str, str]:
    """Return {name: description} for the 30 entities in this subset.

    Picks one canonical row per entity using a known template signature:
      p2d base -> "<name>, known far and wide for being <description>."
      d2p base -> "Q: Who is <description>? A: <name>."
    """
    rows = _read_jsonl(train_path)
    out: dict[str, str] = {}

    if base == "p2d":
        suffix = ", known far and wide for being"
        for r in rows:
            if r["prompt"].endswith(suffix.lstrip(", ")):
                name = r["prompt"][: -len(suffix.lstrip(", "))].rstrip(", ").strip()
                desc = r["completion"].strip().rstrip(".")
                out.setdefault(name, desc)
    elif base == "d2p":
        prefix, suffix = "Q: Who is ", "? A:"
        for r in rows:
            if r["prompt"].startswith(prefix) and r["prompt"].endswith(suffix):
                desc = r["prompt"][len(prefix): -len(suffix)]
                name = r["completion"].strip().rstrip(".")
                out.setdefault(name, desc)
    else:
        raise ValueError(base)

    if len(out) != 30:
        raise RuntimeError(f"{base}: expected 30 entities via canonical template, got {len(out)}")
    return out


def fill_template(template: str, name: str, description: str, *, p2d_template: bool) -> dict:
    """Mirror original_repo/src/tasks/reverse_experiments/reverse_task.py:format_prompt.

    p2d_template -> split at <description>; d2p_template -> split at <name>.
    """
    placeholder    = "<description>" if p2d_template else "<name>"
    split_index    = template.find(placeholder) - 1                       # -1 to keep the leading space on completion
    prompt_tpl     = template[:split_index]
    completion_tpl = template[split_index:]

    def _fill(s: str) -> str:
        return s.replace("<name>", name).replace("<description>", description)

    return {"prompt": _fill(prompt_tpl), "completion": _fill(completion_tpl)}


def generate_reverse_rows(name: str, description: str, rev_templates: list[str],
                          rev_is_p2d: bool, rng: random.Random) -> list[dict]:
    sampled = rng.sample(rev_templates, min(N_REV_PER_ENTITY, len(rev_templates)))
    while len(sampled) < N_REV_PER_ENTITY:
        sampled.append(rng.choice(rev_templates))
    return [fill_template(t, name, description, p2d_template=rev_is_p2d) for t in sampled]


def split_reverse_test(rev_test_rows: list[dict], injected: set[str], base: str) -> tuple[list[dict], list[dict]]:
    """Split <base>_reverse_prompts_test.jsonl by whether the row's entity is in `injected`.

    Direction matters:
      base=p2d -> reverse test asks desc->name; entity name lives in `completion`.
      base=d2p -> reverse test asks name->desc; entity name lives in `prompt`.
    """
    inj, held = [], []
    for r in rev_test_rows:
        if base == "p2d":
            entity_field = r["completion"].strip().rstrip(".").strip()
            matched = entity_field in injected
        else:  # base == "d2p"
            prompt = r["prompt"]
            matched = any(n in prompt for n in injected)
        (inj if matched else held).append(r)
    return inj, held


def build_one(base: str, alpha: int) -> dict:
    """Build one augmented dir for (base, alpha). Returns small summary."""
    rev_base    = "d2p" if base == "p2d" else "p2d"          # opposite direction
    rev_is_p2d  = (rev_base == "p2d")

    base_train  = SRC_DIR / f"{base}_prompts_train.jsonl"
    fwd_test    = SRC_DIR / f"{base}_prompts_test.jsonl"
    rev_test    = SRC_DIR / f"{base}_reverse_prompts_test.jsonl"
    rev_tpl     = TPL_DIR / f"{rev_base}_templates.txt"

    entities    = extract_entities(base_train, base)         # {name: description}, 30 keys
    names       = sorted(entities)                           # deterministic ordering before sampling
    rng         = random.Random(SEED + alpha)                # independent draw per alpha
    k           = math.ceil(alpha / 100 * len(names))
    injected    = set(rng.sample(names, k))

    rev_templates = _load_templates(rev_tpl)
    extra_rows    = []
    for n in sorted(injected):                               # sort -> deterministic row order
        extra_rows.extend(generate_reverse_rows(n, entities[n], rev_templates, rev_is_p2d,
                                                random.Random(SEED + alpha + hash(n) % 10_000)))

    out_dir = OUT_ROOT / f"{base}_aug{alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fwd_train_rows = _read_jsonl(base_train)
    _write_jsonl(out_dir / "train.jsonl", fwd_train_rows + extra_rows)
    _write_jsonl(out_dir / "forward_test.jsonl", _read_jsonl(fwd_test))

    rev_test_rows  = _read_jsonl(rev_test)
    inj_rows, held = split_reverse_test(rev_test_rows, injected, base)
    _write_jsonl(out_dir / "reverse_test_injected.jsonl", inj_rows)
    _write_jsonl(out_dir / "reverse_test_held_out.jsonl", held)

    (out_dir / "injected_entities.json").write_text(
        json.dumps(sorted(injected), indent=2), encoding="utf-8",
    )

    return {
        "base":              base,
        "alpha":             alpha,
        "n_entities":        len(names),
        "n_injected":        k,
        "n_train_total":     len(fwd_train_rows) + len(extra_rows),
        "n_train_orig":      len(fwd_train_rows),
        "n_train_extra":     len(extra_rows),
        "n_rev_inj_test":    len(inj_rows),
        "n_rev_held_test":   len(held),
        "out_dir":           str(out_dir.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    summaries = []
    for base in ("p2d", "d2p"):
        for alpha in ALPHAS:
            s = build_one(base, alpha)
            summaries.append(s)
            print(f"  built {s['out_dir']}: "
                  f"injected={s['n_injected']}/{s['n_entities']}  "
                  f"train={s['n_train_total']} (orig {s['n_train_orig']} + extra {s['n_train_extra']})  "
                  f"rev_test inj/held = {s['n_rev_inj_test']}/{s['n_rev_held_test']}")
    (OUT_ROOT / "_build_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSummary -> {OUT_ROOT / '_build_summary.json'}")


if __name__ == "__main__":
    main()
