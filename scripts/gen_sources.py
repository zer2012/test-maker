"""Étape 1 : génère des modules autonomes avec le teacher -> data/sources.jsonl.

Reprend là où il s'est arrêté : les variations déjà tentées sont sautées.
    python scripts/gen_sources.py --lang python --n 20
"""

import argparse
import ast
import random
import re
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, code_hash, extract_code, load_config, read_jsonl
from testmaker.languages import LANGUAGES
from testmaker.llm import teacher_llm
from testmaker.prompts import DOMAINS, FEATURES, SHAPES, SYNTHETIC_EXTRA, SYNTHETIC_TEMPLATE


def _python_issue(code: str, lang: str) -> str | None:
    """Contrôles locaux et gratuits (Python seulement) avant de payer des tests.

    Rejette le code qui ne compile pas, et celui qui exécute des appels au moment de l'import
    (ex. une démo `demo()` en fin de fichier) : ils tournent avant les tests et peuvent les casser.
    """
    if lang != "python":
        return None
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "ne compile pas"
    for node in tree.body:
        if isinstance(node, ast.If):
            # Seul `if __name__ == "__main__":` est sûr : il ne s'exécute pas quand les tests importent le module.
            if ast.unparse(node.test).replace("'", '"') == '__name__ == "__main__"':
                continue
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
        else:
            calls = [node] if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else []
        if calls:
            return "code exécuté à l'import"
    return None


def _fingerprint(code: str) -> set[str]:
    """Ensemble des mots du code (identifiants, mots-clés) : deux variantes d'un même module en partagent la plupart."""
    return set(re.findall(r"[A-Za-z_]\w+", code))


def _near_duplicate(fp: set[str], others: list[set[str]], threshold: float) -> bool:
    """Vrai si un module déjà gardé partage plus de `threshold` de ses mots avec celui-ci (indice de Jaccard)."""
    return any(len(fp & o) / len(fp | o) >= threshold for o in others if fp | o)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="python", choices=list(LANGUAGES))
    ap.add_argument("--n", type=int, default=20, help="nombre de variations à demander")
    ap.add_argument("--provider")
    args = ap.parse_args()

    cfg = load_config()
    out = cfg["paths"]["sources"]
    spec, llm = LANGUAGES[args.lang], teacher_llm(cfg, args.provider)
    # Reprise : on évite les doublons de code et on ne redemande pas une variation déjà tentée.
    existing = read_jsonl(out)
    seen = {code_hash(r["code"]) for r in existing}
    fingerprints = [_fingerprint(r["code"]) for r in existing]
    tried = {r["id"] for r in existing} | {r["id"] for r in read_jsonl(out + ".rejected")}

    todo = [f"synth-{args.lang}-{i:05d}" for i in range(args.n)]
    todo = [t for t in todo if t not in tried]
    print(f"{len(todo)} modules à générer avec {llm.model}")

    async def one(rid: str) -> tuple[str, str]:
        """Demande un module au teacher ; le numéro de variation choisit le domaine."""
        i = int(rid.rsplit("-", 1)[1])
        # Tirage reproductible (graine = numéro) d'une forme et d'une caractéristique, en plus du domaine.
        rng = random.Random(i)
        prompt = SYNTHETIC_TEMPLATE.format(
            display=spec.display, domain=DOMAINS[i % len(DOMAINS)], seed=i,
            shape=rng.choice(SHAPES), feature=rng.choice(FEATURES),
            extra=SYNTHETIC_EXTRA[args.lang], fence=spec.fence,
        )
        text = await llm.complete([{"role": "user", "content": prompt}],
                                  temperature=cfg["teacher"]["sources_temperature"],
                                  max_tokens=cfg["teacher"]["max_tokens"])
        return rid, text

    kept = 0
    lo, hi = cfg["collect"]["min_lines"], cfg["collect"]["max_lines"]
    # as_completed traite les réponses dans l'ordre où elles arrivent, pas dans l'ordre d'envoi.
    for fut in asyncio.as_completed([one(r) for r in todo]):
        try:
            rid, text = await fut
        except Exception as e:  # une requête perdue ne doit pas arrêter le lot
            print(f"  erreur teacher : {e}")
            continue
        code = spec.normalize(extract_code(text))
        n_lines = len(code.splitlines())
        # Premier motif de rejet rencontré, ou None si le module est gardé.
        reason = _python_issue(code, args.lang) or (
            "pas autonome" if not spec.self_contained(code)
            else f"{n_lines} lignes" if not lo <= n_lines <= hi
            else "doublon" if code_hash(code) in seen
            else "quasi-doublon" if _near_duplicate(_fingerprint(code), fingerprints, 0.6) else None)
        if reason:
            append_jsonl(out + ".rejected", {"id": rid, "reason": reason, "raw": text})
            print(f"  {rid} rejeté ({reason})")
            continue
        seen.add(code_hash(code))
        fingerprints.append(_fingerprint(code))
        append_jsonl(out, {"id": rid, "lang": args.lang, "origin": "synthetic", "code": code})
        kept += 1
        print(f"  {rid} gardé ({n_lines} lignes)")
    print(f"{kept} modules ajoutés à {out}")
    if llm.providers:
        print(f"Fournisseurs : {dict(llm.providers)}")


if __name__ == "__main__":
    asyncio.run(main())
