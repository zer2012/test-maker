"""Élagage des tests : retire les fonctions de test qui échouent sur le code original.

Le teacher écrit souvent 20 à 30 tests dont 1 à 3 ont une valeur attendue fausse. Plutôt que de
jeter tout le fichier, on supprime ces fonctions-là ; le fichier élagué est ensuite revalidé
(original + mutants) comme n'importe quel candidat. Python seulement pour l'instant.
"""

import ast
import re

# Ligne du résumé pytest : `FAILED test_solution.py::TestCart::test_total[3-4] - AssertionError...`
# Groupe 1 = `TestCart::test_total` (sans le fichier ni les paramètres entre crochets).
_SUMMARY = re.compile(r"^(?:FAILED|ERROR) [^\s:]+::([\w:]+)", re.M)
# Erreur de collecte (import impossible, erreur de syntaxe...) : tout le fichier est inutilisable.
_COLLECTION_ERROR = re.compile(r"^ERROR [^\s:]+(?: - .*)?$", re.M)


def failing_tests(log: str) -> set[str] | None:
    """Noms des tests en échec d'après le log pytest ; None si le fichier entier est en erreur."""
    if _COLLECTION_ERROR.search(log):
        return None
    return set(_SUMMARY.findall(log))


def _start(node: ast.AST) -> int:
    """Première ligne d'une définition, décorateurs compris (`@pytest.mark.parametrize`...)."""
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _prune_python(test: str, failing: set[str]) -> tuple[str, int, int] | None:
    """Supprime les fonctions listées dans `failing`. Renvoie (code élagué, retirés, total)."""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    drop: list[tuple[int, int]] = []        # plages de lignes (1-indexées, incluses) à supprimer
    total = removed = 0
    for node in tree.body:
        if isinstance(node, funcs) and node.name.startswith("test"):
            total += 1
            if node.name in failing:
                removed += 1
                drop.append((_start(node), node.end_lineno))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            methods = [m for m in node.body if isinstance(m, funcs) and m.name.startswith("test")]
            bad = [m for m in methods if f"{node.name}::{m.name}" in failing]
            total += len(methods)
            removed += len(bad)
            if bad and len(bad) == len(methods):
                # Une classe sans méthode serait une erreur de syntaxe : on retire la classe entière.
                drop.append((_start(node), node.end_lineno))
            else:
                drop += [(_start(m), m.end_lineno) for m in bad]
    if removed == 0 or removed == total:
        return None
    lines = test.split("\n")
    for start, end in sorted(drop, reverse=True):   # de la fin vers le début : les numéros restent valides
        del lines[start - 1:end]
    pruned = re.sub(r"\n{3,}", "\n\n\n", "\n".join(lines))  # pas de gros trous à la place des tests retirés
    return pruned, removed, total


def prune(lang: str, test: str, log: str, max_ratio: float) -> tuple[str, int] | None:
    """Version élaguée du test, ou None si l'élagage est impossible ou retirerait trop de tests.

    Au-delà de `max_ratio` tests retirés, le teacher a probablement mal compris le code :
    on préfère rejeter le candidat.
    """
    if lang != "python":
        return None
    failing = failing_tests(log)
    if not failing:
        return None
    result = _prune_python(test, failing)
    if result is None:
        return None
    pruned, removed, total = result
    return (pruned, removed) if removed / total <= max_ratio else None
