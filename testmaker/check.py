"""Jugement d'un fichier de test, partagé par la validation (teacher) et l'évaluation (modèles).

Exécute le test sur le code original et les mutants ; s'il échoue partiellement, retire les
fonctions de test fautives (prune.py) et réexécute. Les deux scripts jugent ainsi exactement pareil.
"""

from .prune import prune
from .sandbox import run_tests


def check_test(lang: str, code: str, test: str, muts: list[str], cfg: dict) -> tuple[dict, str]:
    """Renvoie (résultat, version finale du test). Le résultat est celui de la version finale.

    `raw_passed` : le test passait-il tel quel ? `pruned` : nombre de fonctions retirées (0 sinon).
    """
    r = run_tests(lang, code, test, muts, cfg)
    raw_passed, pruned_count = r.passed, 0
    if r.source_ok and not r.passed:
        pruned = prune(lang, test, r.log, cfg["filter"]["max_pruned_ratio"])
        if pruned:
            r2 = run_tests(lang, code, pruned[0], muts, cfg)
            if r2.passed:  # l'élagage n'est retenu que si le reste passe vraiment
                r, test, pruned_count = r2, pruned[0], pruned[1]
    return r.to_dict() | {"raw_passed": raw_passed, "pruned": pruned_count}, test
