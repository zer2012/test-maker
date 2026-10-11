"""Étape 4 : construit le jeu SFT prompt/completion -> data/sft/{train,val,test}.jsonl.

Le split est déterministe (hash de l'id) : le test n'est jamais vu à l'entraînement.
    python scripts/build_sft.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import load_config, read_jsonl, split_of, write_jsonl
from testmaker.prompts import build_messages, format_completion


def main() -> None:
    cfg = load_config()
    threshold = cfg["filter"]["min_mutation_score"]
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for rec in read_jsonl(cfg["paths"]["teacher_val"]):
        split = split_of(rec["id"])
        if split == "test":
            # Pour l'évaluation on garde la source même sans bon test : c'est le modèle évalué qui écrira.
            splits["test"].append({"id": rec["id"], "lang": rec["lang"], "code": rec["code"]})
            continue
        if rec["best"] is None:
            continue
        score = rec["results"][rec["best"]]["mutation_score"]
        # score None = aucun mutant compilable : on ne peut pas juger, on garde.
        if score is not None and score < threshold:
            continue
        splits[split].append({
            "id": rec["id"], "lang": rec["lang"],
            # Format conversationnel prompt/completion de TRL : la perte ne porte que sur la completion.
            "prompt": build_messages(rec["lang"], rec["code"]),
            "completion": [{"role": "assistant",
                            # `tests` contient la version élaguée si elle existe (absent des anciens fichiers).
                            "content": format_completion(rec["lang"], rec.get("tests", rec["candidates"])[rec["best"]])}],
        })
    for name, rows in splits.items():
        write_jsonl(Path(cfg["paths"]["sft_dir"]) / f"{name}.jsonl", rows)
        print(f"{name:5} {len(rows):5}  {dict(Counter(r['lang'] for r in rows))}")


if __name__ == "__main__":
    main()
