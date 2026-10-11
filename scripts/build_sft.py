"""Étape 4 : construit le jeu d'entraînement prompt/completion -> data/sft/{train,val,test}.jsonl.

    python scripts/build_sft.py

Chaque module va dans train (90 %), val (5 %) ou test (5 %) selon le hash de son id : il reste
toujours dans le même split, même quand le dataset grossit. Les fichiers sont réécrits entièrement
à chaque lancement, à partir de data/teacher_val.jsonl.
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import load_config, read_jsonl, split_of, write_jsonl
from testmaker.prompts import build_messages, format_completion

# Une fonction de test pytest (`def test_...`), pour compter les tests d'un fichier.
TEST_FUNCTION = re.compile(r"^\s*(?:async\s+)?def test_", re.M)


def best_test(record: dict) -> str:
    """Le meilleur test du module, dans sa version élaguée si elle existe."""
    # `tests` (versions élaguées) n'existe que dans les fichiers validés après l'ajout de l'élagage.
    return record.get("tests", record["candidates"])[record["best"]]


def rejection_reason(record: dict, cfg: dict) -> str | None:
    """Pourquoi ce module ne donne pas d'exemple d'entraînement, ou None s'il en donne un."""
    if record["best"] is None:
        return "aucun test ne passe"
    score = record["results"][record["best"]]["mutation_score"]
    # score None = aucun mutant compilable : on ne peut pas juger, on garde.
    if score is not None and score < cfg["filter"]["min_mutation_score"]:
        return "score de mutation trop bas"
    # Fichier trop long : le student apprendrait à enchaîner les tests sans jamais s'arrêter.
    if len(TEST_FUNCTION.findall(best_test(record))) > cfg["filter"]["max_test_functions"]:
        return f"plus de {cfg['filter']['max_test_functions']} fonctions de test"
    return None


def training_example(record: dict) -> dict:
    """Un exemple au format conversationnel prompt/completion de TRL (perte calculée sur la completion seule)."""
    return {
        "id": record["id"],
        "lang": record["lang"],
        "prompt": build_messages(record["lang"], record["code"]),     # le même prompt qu'à l'usage
        "completion": [{"role": "assistant", "content": format_completion(record["lang"], best_test(record))}],
    }


def main() -> None:
    cfg = load_config()
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    rejected = Counter()

    for record in read_jsonl(cfg["paths"]["teacher_val"]):
        split = split_of(record["id"])
        if split == "test":
            # Pour l'évaluation, on garde le module même sans bon test : c'est le modèle évalué qui écrira les siens.
            splits["test"].append({"id": record["id"], "lang": record["lang"], "code": record["code"]})
        elif reason := rejection_reason(record, cfg):
            rejected[reason] += 1
        else:
            splits[split].append(training_example(record))

    for name, rows in splits.items():
        write_jsonl(Path(cfg["paths"]["sft_dir"]) / f"{name}.jsonl", rows)
        print(f"{name:5} {len(rows):5}  {dict(Counter(r['lang'] for r in rows))}")
    print(f"modules écartés (train/val) : {dict(rejected)}")


if __name__ == "__main__":
    main()
