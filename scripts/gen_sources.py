"""Étape 1 : génère des modules autonomes avec le teacher -> data/sources.jsonl.

Reprend là où il s'est arrêté : les variations déjà tentées sont sautées.
    python scripts/gen_sources.py --lang python --n 20
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, code_hash, extract_code, load_config, read_jsonl
from testmaker.languages import LANGUAGES
from testmaker.llm import teacher_llm
from testmaker.prompts import DOMAINS, SYNTHETIC_EXTRA, SYNTHETIC_TEMPLATE


def _compiles(code: str, lang: str) -> bool:
    """Contrôle de syntaxe local et gratuit (Python seulement) avant de payer des tests."""
    if lang != "python":
        return True
    try:
        compile(code, "solution.py", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


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
    tried = {r["id"] for r in existing} | {r["id"] for r in read_jsonl(out + ".rejected")}

    todo = [f"synth-{args.lang}-{i:05d}" for i in range(args.n)]
    todo = [t for t in todo if t not in tried]
    print(f"{len(todo)} modules à générer avec {llm.model}")

    async def one(rid: str) -> tuple[str, str]:
        """Demande un module au teacher ; le numéro de variation choisit le domaine."""
        i = int(rid.rsplit("-", 1)[1])
        prompt = SYNTHETIC_TEMPLATE.format(
            display=spec.display, domain=DOMAINS[i % len(DOMAINS)], seed=i,
            extra=SYNTHETIC_EXTRA[args.lang], fence=spec.fence,
        )
        text = await llm.complete([{"role": "user", "content": prompt}],
                                  temperature=0.9, max_tokens=cfg["teacher"]["max_tokens"])
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
        reason = ("ne compile pas" if not _compiles(code, args.lang)
                  else "pas autonome" if not spec.self_contained(code)
                  else f"{n_lines} lignes" if not lo <= n_lines <= hi
                  else "doublon" if code_hash(code) in seen else None)
        if reason:
            append_jsonl(out + ".rejected", {"id": rid, "reason": reason, "raw": text})
            print(f"  {rid} rejeté ({reason})")
            continue
        seen.add(code_hash(code))
        append_jsonl(out, {"id": rid, "lang": args.lang, "origin": "synthetic", "code": code})
        kept += 1
        print(f"  {rid} gardé ({n_lines} lignes)")
    print(f"{kept} modules ajoutés à {out}")
    if llm.providers:
        print(f"Fournisseurs : {dict(llm.providers)}")


if __name__ == "__main__":
    asyncio.run(main())
