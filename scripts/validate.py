"""Étape 3 : exécute chaque candidat dans la sandbox (original + mutants) -> data/teacher_val.jsonl.

Le meilleur candidat est celui qui passe avec le meilleur score de mutation.
    python scripts/validate.py
"""

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, load_config, read_jsonl
from testmaker.mutate import mutants
from testmaker.sandbox import run_tests


def validate(rec: dict, cfg: dict) -> dict:
    """Exécute tous les candidats d'une source contre les mêmes mutants et choisit le meilleur."""
    muts = mutants(rec["code"], rec["lang"], cfg["sandbox"]["n_mutants"], seed=rec["id"])
    results = [run_tests(rec["lang"], rec["code"], test, muts, cfg).to_dict()
               for test in rec["candidates"]]
    passing = [i for i, r in enumerate(results) if r["passed"]]
    # Meilleur = plus haut score de mutation ; à égalité, le test le plus court. None si aucun ne passe.
    best = max(passing, key=lambda i: (results[i]["mutation_score"] or 0,
                                       -len(rec["candidates"][i])), default=None)
    return rec | {"results": results, "best": best}


def main() -> None:
    cfg = load_config()
    src, out = cfg["paths"]["teacher_gen"], cfg["paths"]["teacher_val"]
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(src) if r["id"] not in done]
    print(f"{len(todo)} sources à valider")

    # Des threads suffisent : le vrai travail est fait par les conteneurs Docker, en parallèle.
    with ThreadPoolExecutor(cfg["sandbox"]["workers"]) as pool:
        futs = [pool.submit(validate, r, cfg) for r in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            append_jsonl(out, rec)
            summary = " ".join(
                "BUILD_FAIL" if not r["source_ok"] else
                f"pass({r['mutants_killed']}/{r['mutants_total']})" if r["passed"] else "fail"
                for r in rec["results"])
            print(f"  [{k}/{len(todo)}] {rec['id']} : {summary}  best={rec['best']}")

    report(read_jsonl(out), cfg)


def report(records: list[dict], cfg: dict) -> None:
    """Affiche le bilan global de la validation (tous les enregistrements, pas seulement ce lancement)."""
    threshold = cfg["filter"]["min_mutation_score"]
    stats = Counter()
    for rec in records:
        stats["sources"] += 1
        for r in rec["results"]:
            stats["candidats"] += 1
            stats["candidats qui passent"] += r["passed"]
        if rec["best"] is not None:
            score = rec["results"][rec["best"]]["mutation_score"]
            stats["sources avec un test qui passe"] += 1
            stats[f"sources gardées (mutation >= {threshold})"] += score is None or score >= threshold
    print("\n--- Bilan ---")
    for k, v in stats.items():
        print(f"{k:40} {v}")


if __name__ == "__main__":
    main()
