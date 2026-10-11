"""Conventions par langage : fichiers, framework de test, image Docker, heuristiques d'autonomie.

Chaque langage définit comment disposer le code et le test dans un dossier, comment les
compiler/exécuter, et ce que le modèle doit savoir de cette disposition (le `hint`).
Le même hint est utilisé pour le teacher, l'entraînement et l'inférence : c'est
l'« interface » que TestMaker apprend.
"""

import re
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Language:
    """Fiche d'un langage : tout ce dont le prompt, la collecte et la sandbox ont besoin."""

    name: str                               # identifiant interne ("python")
    display: str                            # nom affiché dans le prompt ("Python")
    exts: tuple[str, ...]                   # extensions de fichiers reconnues à la collecte
    fence: str                              # langage des blocs ```fence
    framework: str                          # framework de test imposé dans le prompt
    image: str                              # image Docker de la sandbox
    build_cmd: str                          # compile le code seul ; `{code_file}` est substitué
    test_cmd: str                           # compile et exécute les tests
    code_file: Callable[[str], str]
    files: Callable[[str, str], dict]       # (code, test) -> {chemin: contenu}
    hint: Callable[[str], str]
    normalize: Callable[[str], str]         # appliqué une fois à la collecte
    self_contained: Callable[[str], bool]


def _lines(code: str) -> list[str]:
    """Découpe le code en lignes."""
    return code.splitlines()


# --- Python -----------------------------------------------------------------

_STDLIB = set(sys.stdlib_module_names)


def _py_self_contained(code: str) -> bool:
    """Vrai si le code définit au moins une fonction, n'importe que la stdlib et ne lit pas stdin."""
    if not re.search(r"^\s*def \w+", code, re.M):
        return False
    # Groupe 1 : `from X.y import ...` ; groupe 2 : `import a, b.c as d`.
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", code, re.M):
        mods = [m.group(1)] if m.group(1) else m.group(2).split(",")
        for mod in mods:
            top = mod.strip().split(" ")[0].split(".")[0]   # "os.path as p" -> "os"
            if top and top not in _STDLIB:
                return False
    return not re.search(r"\binput\(", code)


PYTHON = Language(
    name="python", display="Python", exts=(".py",), fence="python", framework="pytest",
    image="testmaker-python",
    build_cmd="python -m py_compile {code_file}",
    test_cmd="python -m pytest -q -x -p no:cacheprovider test_solution.py",
    code_file=lambda code: "solution.py",
    files=lambda code, test: {"solution.py": code, "test_solution.py": test},
    hint=lambda code: "The code lives in `solution.py`; import from it with `from solution import ...`. "
                      "Write the test file `test_solution.py`.",
    normalize=lambda code: code,
    self_contained=_py_self_contained,
)


# --- JavaScript (CommonJS + node:test, aucune dépendance) ---------------------

_NODE_BUILTINS = {"assert", "buffer", "crypto", "events", "fs", "os", "path", "querystring",
                  "stream", "string_decoder", "url", "util", "zlib"}


def _js_normalize(code: str) -> str:
    """Ajoute un module.exports pour que les tests puissent importer les symboles de premier niveau."""
    if re.search(r"\bmodule\.exports\b|\bexports\.\w+", code):
        return code
    # Noms de premier niveau (en début de ligne, donc pas imbriqués) : fonctions, classes, const/let/var.
    found = re.findall(
        r"^(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"
        r"|^class\s+([A-Za-z_$][\w$]*)"
        r"|^(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=",
        code, re.M,
    )
    # findall renvoie un tuple de 3 groupes par match, dont un seul est rempli.
    names = sorted({n for group in found for n in group if n})
    if not names:
        return code
    return code.rstrip() + "\n\nmodule.exports = { " + ", ".join(names) + " };\n"


def _js_self_contained(code: str) -> bool:
    """Vrai pour du CommonJS sans ESM, sans DOM et sans `require` hors modules Node intégrés."""
    if re.search(r"^\s*(import|export)\s", code, re.M):
        return False
    if re.search(r"\b(document|window|jQuery|localStorage)\b|\$\(", code):
        return False
    for mod in re.findall(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        if mod.removeprefix("node:") not in _NODE_BUILTINS:
            return False
    return bool(re.search(r"\bfunction\b|=>", code))


JAVASCRIPT = Language(
    name="javascript", display="JavaScript", exts=(".js",), fence="javascript",
    framework="node:test with node:assert/strict", image="testmaker-node",
    build_cmd="node --check {code_file}",
    test_cmd="node --test solution.test.js",
    code_file=lambda code: "solution.js",
    files=lambda code, test: {"solution.js": code, "solution.test.js": test},
    hint=lambda code: "The code lives in `solution.js` (CommonJS); import it with "
                      "`const { ... } = require('./solution');`. Use only `node:test` and "
                      "`node:assert/strict`. Write the test file `solution.test.js`.",
    normalize=_js_normalize,
    self_contained=_js_self_contained,
)


# --- Go -------------------------------------------------------------------------

def _go_package(code: str) -> str:
    """Nom du package Go déclaré (le test doit être dans le même)."""
    m = re.search(r"^package\s+(\w+)", code, re.M)
    return m.group(1) if m else "main"


def _go_self_contained(code: str) -> bool:
    """Vrai si seuls des packages standard sont importés (pas de point dans le 1er segment, ex. github.com)."""
    if not re.search(r"^package\s+\w+", code, re.M) or "func " not in code:
        return False
    # Lignes `import "x"`, `import alias "x"` ou `"x"` dans un bloc import ( ... ).
    imports = re.findall(r'^\s*(?:import\s+)?(?:\w+\s+)?"([^"]+)"', code, re.M)
    return all("." not in path.split("/")[0] for path in imports)


GO = Language(
    name="go", display="Go", exts=(".go",), fence="go", framework="the standard `testing` package",
    image="testmaker-go",
    build_cmd="go build -o /dev/null ./...",
    test_cmd="go test ./...",
    code_file=lambda code: "solution.go",
    files=lambda code, test: {"go.mod": "module sol\n\ngo 1.22\n",
                              "solution.go": code, "solution_test.go": test},
    hint=lambda code: f"The code lives in `solution.go` in package `{_go_package(code)}`. "
                      "Write `solution_test.go` in the same package using only the standard library.",
    normalize=lambda code: code,
    self_contained=_go_self_contained,
)


# --- Rust ---------------------------------------------------------------------

def _rust_self_contained(code: str) -> bool:
    """Vrai si le code n'a ni tests existants, ni module externe (`mod x;`), ni crate tierce."""
    if "fn " not in code or "#[cfg(test)]" in code or "extern crate" in code:
        return False
    if re.search(r"^\s*(pub\s+)?mod\s+\w+\s*;", code, re.M):
        return False
    roots = re.findall(r"^\s*(?:pub\s+)?use\s+:?:?(\w+)", code, re.M)
    return all(r in {"std", "core", "alloc", "crate", "self", "super"} for r in roots)


RUST = Language(
    name="rust", display="Rust", exts=(".rs",), fence="rust", framework="built-in #[test]",
    image="testmaker-rust",
    build_cmd="cargo build --offline -q",
    test_cmd="cargo test --offline -q",
    code_file=lambda code: "src/lib.rs",
    files=lambda code, test: {
        "Cargo.toml": '[package]\nname = "sol"\nversion = "0.1.0"\nedition = "2021"\n\n[lib]\npath = "src/lib.rs"\n',
        "src/lib.rs": code.rstrip() + "\n\n" + test + "\n",
    },
    hint=lambda code: "The code is `src/lib.rs` of a library crate named `sol`. Write a "
                      "`#[cfg(test)] mod tests { use super::*; ... }` module that will be appended "
                      "to the end of that file. Use only the standard library.",
    normalize=lambda code: code,
    self_contained=_rust_self_contained,
)


# --- Java (JUnit 5) -------------------------------------------------------------

def _java_file(code: str) -> str:
    """Java impose fichier = classe publique : on déduit le nom du fichier de la classe."""
    m = re.search(r"^public\s+(?:(?:final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+(\w+)",
                  code, re.M)
    return f"{m.group(1)}.java" if m else "Solution.java"


def _java_self_contained(code: str) -> bool:
    """Vrai si le code contient une classe/record et n'importe que `java.*`."""
    if "class " not in code and "record " not in code:
        return False
    imports = re.findall(r"^\s*import\s+(?:static\s+)?([\w.]+)", code, re.M)
    return all(i.startswith("java.") for i in imports)


JAVA = Language(
    name="java", display="Java", exts=(".java",), fence="java", framework="JUnit 5",
    image="testmaker-java",
    build_cmd="mkdir -p /tmp/b && javac -d /tmp/b {code_file}",
    test_cmd="mkdir -p out && javac -d out -cp /opt/junit.jar *.java && "
             "java -jar /opt/junit.jar execute -cp out --select-class GeneratedTest "
             "--fail-if-no-tests --disable-banner --details=summary",
    code_file=_java_file,
    files=lambda code, test: {_java_file(code): code, "GeneratedTest.java": test},
    hint=lambda code: f"The code lives in `{_java_file(code)}` in the default package. Write a JUnit 5 "
                      "test class named `GeneratedTest` (file `GeneratedTest.java`, no package "
                      "declaration) using `org.junit.jupiter.api` and `Assertions`.",
    normalize=lambda code: re.sub(r"^\s*package\s+[\w.]+\s*;\s*\n", "", code, flags=re.M),
    self_contained=_java_self_contained,
)


# Registre : nom -> fiche, et extension -> fiche (pour la collecte de fichiers).
LANGUAGES: dict[str, Language] = {l.name: l for l in (PYTHON, JAVASCRIPT, GO, RUST, JAVA)}
BY_EXT = {ext: l for l in LANGUAGES.values() for ext in l.exts}
