"""Étape 1 : le teacher invente des modules de code autonomes à tester -> data/sources.jsonl.

    python scripts/gen_sources.py --lang python --n 500 --provider openrouter

Chaque module porte un id `synth-<langage>-<numéro>`. Le numéro fixe le domaine, la forme et la
caractéristique demandés : relancer avec un --n plus grand ne demande que les numéros jamais tentés.
Les modules rejetés vont dans data/sources.jsonl.rejected, avec la raison du rejet.
"""

import argparse
import ast
import asyncio
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, code_hash, extract_code, load_config, read_jsonl
from testmaker.languages import LANGUAGES, Language
from testmaker.llm import LLM, teacher_llm
from testmaker.prompts import DOMAINS, FEATURES, SHAPES, SYNTHETIC_EXTRA, SYNTHETIC_TEMPLATE

# Deux modules qui partagent au moins 60 % de leurs mots sont considérés comme des variantes l'un de l'autre.
NEAR_DUPLICATE_THRESHOLD = 0.6


# --- Le prompt --------------------------------------------------------------------------------

def build_prompt(number: int, spec: Language) -> str:
    """Demande d'invention du module n° `number` : domaine, forme et caractéristique en dépendent."""
    # Graine = numéro : le module 48 reçoit toujours la même combinaison, même si on le redemande.
    rng = random.Random(number)
    return SYNTHETIC_TEMPLATE.format(
        display=spec.display,
        domain=DOMAINS[number % len(DOMAINS)],   # les 40 domaines défilent en boucle
        seed=number,
        shape=rng.choice(SHAPES),
        feature=rng.choice(FEATURES),
        extra=SYNTHETIC_EXTRA[spec.name],
        fence=spec.fence,
    )


# --- Les contrôles avant de garder un module ----------------------------------------------------

def python_issue(code: str) -> str | None:
    """Contrôles gratuits propres à Python : le code compile-t-il, exécute-t-il des choses à l'import ?

    Un appel au niveau du module (ex. une démo `demo()` en fin de fichier) s'exécute quand les tests
    font `from solution import ...` : s'il plante, tous les tests échouent avant même de commencer.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "ne compile pas"
    for node in tree.body:                      # uniquement les instructions de premier niveau
        if isinstance(node, ast.If):
            # Seul `if __name__ == "__main__":` est sûr : ce bloc ne s'exécute pas à l'import.
            if ast.unparse(node.test).replace("'", '"') == '__name__ == "__main__"':
                continue
            has_call = any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in ast.walk(node))
        else:
            has_call = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        if has_call:
            return "code exécuté à l'import"
    return None


def fingerprint(code: str) -> set[str]:
    """Ensemble des mots du code (noms, mots-clés) : deux variantes d'un même module en partagent la plupart."""
    return set(re.findall(r"[A-Za-z_]\w+", code))


class KnownModules:
    """Les modules déjà gardés, pour repérer les doublons exacts et les quasi-doublons."""

    def __init__(self, existing: list[dict]):
        self.hashes = {code_hash(r["code"]) for r in existing}
        self.fingerprints = [fingerprint(r["code"]) for r in existing]

    def is_duplicate(self, code: str) -> bool:
        return code_hash(code) in self.hashes

    def is_near_duplicate(self, code: str) -> bool:
        # Indice de Jaccard : mots en commun / mots au total, comparé à chaque module déjà gardé.
        fp = fingerprint(code)
        return any(len(fp & other) / len(fp | other) >= NEAR_DUPLICATE_THRESHOLD
                   for other in self.fingerprints if fp | other)

    def add(self, code: str) -> None:
        self.hashes.add(code_hash(code))
        self.fingerprints.append(fingerprint(code))


def rejection_reason(code: str, spec: Language, known: KnownModules, cfg: dict) -> str | None:
    """Première raison de rejeter le module, ou None s'il est gardé. Les contrôles vont du moins cher au plus cher."""
    if spec.name == "python" and (issue := python_issue(code)):
        return issue
    if not spec.self_contained(code):          # dépendances hors bibliothèque standard, I/O...
        return "pas autonome"
    n_lines = len(code.splitlines())
    if not cfg["collect"]["min_lines"] <= n_lines <= cfg["collect"]["max_lines"]:
        return f"{n_lines} lignes"
    if known.is_duplicate(code):
        return "doublon"
    if known.is_near_duplicate(code):
        return "quasi-doublon"
    return None


# --- Programme principal ----------------------------------------------------------------------

def ids_to_generate(lang: str, n: int, out: str) -> list[str]:
    """Ids de 0 à n-1 jamais tentés : ni gardés, ni rejetés lors d'un lancement précédent."""
    tried = {r["id"] for r in read_jsonl(out)} | {r["id"] for r in read_jsonl(out + ".rejected")}
    return [rid for i in range(n) if (rid := f"synth-{lang}-{i:05d}") not in tried]


async def ask_teacher(llm: LLM, rid: str, spec: Language, cfg: dict) -> tuple[str, str]:
    """Demande le module `rid` au teacher. Renvoie (id, réponse brute)."""
    number = int(rid.rsplit("-", 1)[1])
    answer = await llm.complete([{"role": "user", "content": build_prompt(number, spec)}],
                                temperature=cfg["teacher"]["sources_temperature"],
                                max_tokens=cfg["teacher"]["max_tokens"])
    return rid, answer


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="python", choices=list(LANGUAGES))
    ap.add_argument("--n", type=int, default=20, help="numéros de module à couvrir (0 à n-1)")
    ap.add_argument("--provider", help="profil de teacher (config.yaml), sinon teacher.provider")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["paths"]["sources"]
    spec, llm = LANGUAGES[args.lang], teacher_llm(cfg, args.provider)
    known = KnownModules(read_jsonl(out))
    todo = ids_to_generate(args.lang, args.n, out)
    print(f"{len(todo)} modules à générer avec {llm.model}")

    kept = 0
    # as_completed : on traite les réponses dans l'ordre où elles arrivent, pas dans l'ordre d'envoi.
    for finished in asyncio.as_completed([ask_teacher(llm, rid, spec, cfg) for rid in todo]):
        try:
            rid, answer = await finished
        except Exception as e:  # une requête perdue ne doit pas arrêter le lot ; elle sera retentée à la relance
            print(f"  erreur teacher : {e}")
            continue
        code = spec.normalize(extract_code(answer))
        if reason := rejection_reason(code, spec, known, cfg):
            append_jsonl(out + ".rejected", {"id": rid, "reason": reason, "raw": answer})
            print(f"  {rid} rejeté ({reason})")
            continue
        known.add(code)
        append_jsonl(out, {"id": rid, "lang": args.lang, "origin": "synthetic", "code": code})
        kept += 1
        print(f"  {rid} gardé ({len(code.splitlines())} lignes)")

    print(f"{kept} modules ajoutés à {out}")
    if llm.providers:
        print(f"Fournisseurs : {dict(llm.providers)}")


if __name__ == "__main__":
    asyncio.run(main())
