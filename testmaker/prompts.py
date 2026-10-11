"""Prompts envoyés aux modèles.

Le prompt de génération de tests (`build_messages`) est l'interface de TestMaker : il doit être
identique pour le teacher, l'entraînement SFT et l'inférence du student. Le modifier après
l'entraînement dégrade le modèle. Le prompt synthétique ne sert qu'à fabriquer des sources.
"""

from .languages import LANGUAGES

# Rôle système commun à tous les langages.
SYSTEM = (
    "You are TestMaker, an expert at writing thorough, correct and runnable unit tests "
    "in many programming languages. You test observable behaviour: normal cases, edge "
    "cases and error handling. Your tests pass against the code as written."
)

# Demande de tests ; les {champs} viennent de la fiche du langage (languages.py).
USER_TEMPLATE = """Write unit tests for the following {display} code.

- Test framework: {framework}
- {hint}
- Cover normal cases, edge cases and error handling; assert on concrete values.
- Write 8 to 20 focused test functions; each one checks a single behaviour with few assertions.
- Test the public behaviour only: do not inspect private attributes or helpers (names starting with `_`).
- Compare floating-point results with a tolerance, never with exact equality.
- Do not modify or repeat the code under test.
- Answer with a single ```{fence} code block containing the complete test code, nothing else.

`{code_file}`:
```{fence}
{code}
```"""


def build_messages(lang: str, code: str) -> list[dict]:
    """Construit la conversation (system + user) qui demande des tests pour `code`."""
    spec = LANGUAGES[lang]
    user = USER_TEMPLATE.format(
        display=spec.display, framework=spec.framework, hint=spec.hint(code),
        fence=spec.fence, code_file=spec.code_file(code), code=code.rstrip(),
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def format_completion(lang: str, test_code: str) -> str:
    """Réponse cible du SFT : le test dans un bloc de code, exactement le format demandé par le prompt."""
    return f"```{LANGUAGES[lang].fence}\n{test_code.strip()}\n```"


# Demande au teacher d'inventer un module à tester (étape 1, sources synthétiques).
# `seed` rend chaque requête différente ; `extra` ajoute les contraintes propres au langage.
SYNTHETIC_TEMPLATE = """Write a realistic, self-contained {display} module of 40 to 150 lines for this domain: {domain}.
Invent a concrete, specific task (variation #{seed}). Shape: {shape}. It must also feature {feature}.
Include real logic: branches, loops, input validation and error cases. Avoid the most obvious textbook example.
Constraints: standard library only, no I/O, no network, no reading from stdin, no main entry point.
{extra}
Answer with a single ```{fence} code block and nothing else."""

# Deux axes de plus, combinés au domaine (40 x 8 x 10 = 3 200 combinaisons) : sans eux, 500 modules
# sur 40 domaines donnent ~12 variantes très proches par domaine (12 caches LRU...).
SHAPES = [
    "a small class with internal state and 4-6 methods",
    "4-6 independent pure functions",
    "a parser that turns text into structured data, plus helpers",
    "a validator that raises custom error types with clear messages",
    "a small data structure with insert, remove and query operations",
    "an algorithm split into a main function and several helpers",
    "two cooperating classes (e.g. a container and its items)",
    "a calculator/engine applying configurable rules",
]
FEATURES = [
    "custom error types", "recursion", "a fixed set of named constants or an enum",
    "immutable value objects", "nested collections", "string building and formatting",
    "sorting with custom keys", "guard clauses and early returns", "numeric edge cases (zero, negative, rounding)",
    "optional parameters with defaults",
]

# Contraintes qui rendent le module compatible avec la disposition de fichiers de languages.py.
SYNTHETIC_EXTRA = {
    "python": "",
    "javascript": "Use CommonJS and end with `module.exports = { ... }`.",
    "go": "Use `package solution`.",
    "rust": "Write it as the content of a library `lib.rs`; make the public API `pub`. No tests.",
    "java": "No package declaration. Exactly one public top-level class.",
}

# Domaines parcourus en rotation (variation i -> DOMAINS[i % len]) pour diversifier les sources.
DOMAINS = [
    "string parsing and formatting", "date and time arithmetic", "money and currency calculations",
    "inventory management", "matrix operations", "graph algorithms", "text tokenization",
    "rate limiting", "caching with eviction", "URL and query-string handling", "CSV-like parsing",
    "validation of user input", "geometry", "statistics", "priority queues and scheduling",
    "state machines", "unit conversion", "bank accounts and transactions", "shopping cart pricing",
    "sorting and searching", "roman numerals and number bases", "configuration merging",
    "event sourcing", "tree data structures", "interval arithmetic", "password policy checks",
    "simple expression evaluation", "recursive algorithms", "bit manipulation", "game rules (board or cards)",
    "semantic version comparison", "text diffing", "run-length and other simple encodings",
    "path manipulation", "retry and backoff policies", "permissions and roles", "tax computation",
    "calendar and scheduling conflicts", "LRU/LFU structures", "string similarity metrics",
]
