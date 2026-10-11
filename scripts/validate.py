"""Étape 3 : exécute chaque candidat dans la sandbox (original + mutants) -> data/teacher_val.jsonl.

Un candidat dont quelques tests échouent est élagué (ces fonctions sont retirées) puis revalidé.
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
from testmaker.prune import prune
from testmaker.sandbox import run_tests


def validate(rec: dict, cfg: dict) -> dict:
    """Exécute tous les candidats d'une source contre les mêmes mutants et choisit le meilleur."""
    lang, code = rec["lang"], rec["code"]
    muts = mutants(code, lang, cfg["sandbox"]["n_mutants"], seed=rec["id"])
    results, tests = [], []      # tests[i] = version finale du candidat i (élaguée ou non)
    for test in rec["candidates"]:
        r = run_tests(lang, code, test, muts, cfg)
        pruned_count = 0
        if r.source_ok and not r.passed:
            pruned = prune(lang, test, r.log, cfg["filter"]["max_pruned_ratio"])
            if pruned:
                r2 = run_tests(lang, code, pruned[0], muts, cfg)
                if r2.passed:  # l'élagage n'est retenu que si le reste passe vraiment
                    r, test, pruned_count = r2, pruned[0], pruned[1]
        results.append(r.to_dict() | {"pruned": pruned_count})
        tests.append(test)
    passing = [i for i, r in enumerate(results) if r["passed"]]
    # Meilleur = plus haut score de mutation ; à égalité, le test le plus court. None si aucun ne passe.
    best = max(passing, key=lambda i: (results[i]["mutation_score"] or 0, -len(tests[i])), default=None)
    return rec | {"results": results, "tests": tests, "best": best}


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
                f"pass({r['mutants_killed']}/{r['mutants_total']}"
                + (f", -{r['pruned']} tests)" if r.get("pruned") else ")") if r["passed"] else "fail"
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
            stats["  dont après élagage"] += r["passed"] and bool(r.get("pruned"))
        if rec["best"] is not None:
            score = rec["results"][rec["best"]]["mutation_score"]
            stats["sources avec un test qui passe"] += 1
            stats[f"sources gardées (mutation >= {threshold})"] += score is None or score >= threshold
    print("\n--- Bilan ---")
    for k, v in stats.items():
        print(f"{k:40} {v}")


if __name__ == "__main__":
    main()
