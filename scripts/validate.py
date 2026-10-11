"""Étape 3 : exécute chaque candidat dans la sandbox Docker -> data/teacher_val.jsonl.

    python scripts/validate.py

Pour chaque module : chaque candidat est exécuté sur le code original puis sur des mutants
(copies du code avec un bug introduit). Un candidat dont quelques tests échouent est élagué
(fonctions fautives retirées) puis revalidé. Le meilleur candidat est celui qui passe avec le
meilleur score de mutation. Les modules déjà validés sont sautés.
"""

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.check import check_test
from testmaker.common import append_jsonl, load_config, read_jsonl
from testmaker.mutate import mutants


def best_candidate(results: list[dict], tests: list[str]) -> int | None:
    """Index du meilleur candidat : celui qui passe avec le plus haut score de mutation.

    À égalité, le test le plus court (plus simple à apprendre). None si aucun candidat ne passe.
    """
    passing = [i for i, r in enumerate(results) if r["passed"]]
    return max(passing, key=lambda i: (results[i]["mutation_score"] or 0, -len(tests[i])), default=None)


def validate(record: dict, cfg: dict) -> dict:
    """Exécute tous les candidats d'un module contre les mêmes mutants. Renvoie le record enrichi."""
    lang, code = record["lang"], record["code"]
    # Graine = id du module : mêmes mutants pour tous ses candidats, et à chaque relance.
    muts = mutants(code, lang, cfg["sandbox"]["n_mutants"], seed=record["id"])
    results, tests = [], []
    for candidate in record["candidates"]:
        result, final_test = check_test(lang, code, candidate, muts, cfg)   # exécution + élagage éventuel
        results.append(result)
        tests.append(final_test)          # version élaguée si l'élagage a été retenu, sinon l'originale
    return record | {"results": results, "tests": tests, "best": best_candidate(results, tests)}


def describe(record: dict) -> str:
    """Résumé d'une ligne : le verdict de chaque candidat, puis l'index du meilleur."""
    verdicts = []
    for r in record["results"]:
        if not r["source_ok"]:
            verdicts.append("BUILD_FAIL")       # le code original lui-même ne compile pas
        elif not r["passed"]:
            verdicts.append("fail")
        else:
            pruned = f", -{r['pruned']} tests" if r.get("pruned") else ""
            verdicts.append(f"pass({r['mutants_killed']}/{r['mutants_total']}{pruned})")
    return f"{' '.join(verdicts)}  best={record['best']}"


def report(records: list[dict], cfg: dict) -> None:
    """Bilan global de la validation (tous les modules validés, pas seulement ceux de ce lancement)."""
    threshold = cfg["filter"]["min_mutation_score"]
    stats = Counter()
    for record in records:
        stats["sources"] += 1
        for r in record["results"]:
            stats["candidats"] += 1
            stats["candidats qui passent"] += r["passed"]
            stats["  dont après élagage"] += r["passed"] and bool(r.get("pruned"))
        if record["best"] is not None:
            score = record["results"][record["best"]]["mutation_score"]
            stats["sources avec un test qui passe"] += 1
            # Même règle que build_sft : score None (aucun mutant compilable) = on ne peut pas juger, on garde.
            stats[f"sources gardées (mutation >= {threshold})"] += score is None or score >= threshold
    print("\n--- Bilan ---")
    for label, value in stats.items():
        print(f"{label:40} {value}")


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["teacher_val"]
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(cfg["paths"]["teacher_gen"]) if r["id"] not in done]
    print(f"{len(todo)} sources à valider")

    # Des threads suffisent : le vrai travail est fait par les conteneurs Docker, en parallèle.
    with ThreadPoolExecutor(cfg["sandbox"]["workers"]) as pool:
        futures = [pool.submit(validate, record, cfg) for record in todo]
        for k, future in enumerate(as_completed(futures), 1):
            record = future.result()
            append_jsonl(out, record)     # écrit au fil de l'eau : un arrêt ne perd que les modules en cours
            print(f"  [{k}/{len(todo)}] {record['id']} : {describe(record)}")

    report(read_jsonl(out), cfg)


if __name__ == "__main__":
    main()
