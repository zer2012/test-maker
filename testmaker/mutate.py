"""Mutation testing léger : un bon jeu de tests doit échouer quand on altère le code.

Sert à écarter les tests triviaux (`assert True`, tests qui n'appellent rien...) qui passent
toujours. Les mutants qui ne compilent plus sont éliminés par la sandbox (étape build).
"""

import random
import re

# Opérateurs de mutation : (motif regex, remplacement). Les lookbehind `(?<!...)` et
# lookahead `(?!...)` évitent de couper un opérateur plus long : par exemple `==` ne doit
# pas matcher à l'intérieur de `===` (JS), et `<=` ne doit pas matcher dans `<<=`.
COMMON_OPS = [
    (r"(?<![=!<>])==(?!=)", "!="),
    (r"(?<![=!])!=(?!=)", "=="),
    (r"(?<![<>=!])<=", ">"),
    (r"(?<![<>=!-])>=", "<"),           # `-` exclu pour ne pas toucher `->=`
    (r"(?<=\s)<(?=\s)", ">="),          # `<` entouré d'espaces : évite les génériques `Vec<T>`
    (r"(?<=\s)>(?=\s)", "<="),
    (r" \+ ", " - "),                   # espaces exigés : évite `++`, `+=` et les signes unaires
    (r" - ", " + "),
    (r" \* ", " / "),
    (r"(?<![\w.])1(?![\w.])", "0"),     # le chiffre 1 isolé (pas dans `x1`, `10` ou `1.5`)
    (r"(?<![\w.])0(?![\w.])", "1"),
]
LANG_OPS = {
    "python": [(r"\band\b", "or"), (r"\bor\b", "and"), (r"\bTrue\b", "False"),
               (r"\bFalse\b", "True"), (r"\bnot ", "")],
    "default": [(r"&&", "||"), (r"\|\|", "&&"), (r"\btrue\b", "false"), (r"\bfalse\b", "true")],
}
# Lignes jamais mutées : commentaires, imports, exports, annotations/décorateurs.
SKIP_PREFIXES = ("#", "//", "/*", "*", "import ", "from ", "use ", "package ", "module.exports", "@")


def _in_string(line: str, pos: int) -> bool:
    """Heuristique : la position `pos` est-elle dans une chaîne littérale ?

    On compte les guillemets avant `pos` : un nombre impair signifie qu'une chaîne est ouverte.
    Les caractères échappés (`\\"`) sont retirés d'abord pour ne pas fausser le compte.
    """
    prefix = re.sub(r"\\.", "", line[:pos])
    return prefix.count('"') % 2 == 1 or prefix.count("'") % 2 == 1 or prefix.count("`") % 2 == 1


def mutants(code: str, lang: str, k: int, seed: str) -> list[str]:
    """Renvoie jusqu'à k variantes du code, chacune avec une seule petite modification.

    `seed` (l'id de la source) rend le tirage reproductible : relancer la validation
    produit les mêmes mutants, donc des scores comparables entre candidats.
    """
    ops = COMMON_OPS + LANG_OPS.get(lang, LANG_OPS["default"])
    lines = code.split("\n")
    # 1) Recense tous les « sites » mutables : (ligne, début, fin, remplacement).
    sites, in_docstring = [], False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Un nombre impair de `"""` sur la ligne ouvre ou ferme une docstring Python.
        if lang == "python" and (stripped.count('"""') + stripped.count("'''")) % 2 == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring or not stripped or stripped.startswith(SKIP_PREFIXES):
            continue
        for pattern, repl in ops:
            for m in re.finditer(pattern, line):
                if not _in_string(line, m.start()):
                    sites.append((i, m.start(), m.end(), repl))

    # 2) Mélange déterministe, puis applique les sites un par un jusqu'à obtenir k mutants distincts.
    random.Random(seed).shuffle(sites)
    out: list[str] = []
    for i, start, end, repl in sites:
        new_lines = list(lines)
        new_lines[i] = lines[i][:start] + repl + lines[i][end:]
        candidate = "\n".join(new_lines)
        if candidate != code and candidate not in out:
            out.append(candidate)
        if len(out) == k:
            break
    return out
