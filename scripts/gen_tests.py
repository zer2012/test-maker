"""Étape 2 : le teacher écrit plusieurs fichiers de test candidats par module -> data/teacher_gen.jsonl.

    python scripts/gen_tests.py --lang python --provider openrouter
    python scripts/gen_tests.py --limit 10          # seulement 10 modules, pour essayer

Les modules qui ont déjà leurs candidats sont sautés : une relance reprend là où on s'est arrêté.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, extract_code, load_config, read_jsonl
from testmaker.llm import LLM, teacher_llm
from testmaker.prompts import build_messages


def modules_to_do(cfg: dict, lang: str | None, limit: int | None) -> list[dict]:
    """Modules de sources.jsonl qui n'ont pas encore de candidats (filtrés par langage si demandé)."""
    done = {r["id"] for r in read_jsonl(cfg["paths"]["teacher_gen"])}
    todo = [m for m in read_jsonl(cfg["paths"]["sources"])
            if m["id"] not in done and (lang is None or m["lang"] == lang)]
    return todo[:limit]


async def ask_candidates(llm: LLM, module: dict, cfg: dict) -> tuple[dict, list[str]]:
    """Demande `n_candidates` fichiers de test pour un module. Renvoie (module, réponses brutes)."""
    t = cfg["teacher"]
    # Le prompt est celui que le student verra à l'entraînement et à l'usage (prompts.USER_TEMPLATE).
    answers = await llm.sample(build_messages(module["lang"], module["code"]), n=t["n_candidates"],
                               temperature=t["temperature"], max_tokens=t["max_tokens"])
    return module, answers


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", help="ne traiter que ce langage")
    ap.add_argument("--limit", type=int, help="ne traiter que N modules")
    ap.add_argument("--provider", help="profil de teacher (config.yaml), sinon teacher.provider")
    args = ap.parse_args()

    cfg = load_config()
    llm = teacher_llm(cfg, args.provider)
    todo = modules_to_do(cfg, args.lang, args.limit)
    print(f"{len(todo)} sources x {cfg['teacher']['n_candidates']} candidats avec {llm.model}")

    for k, finished in enumerate(asyncio.as_completed([ask_candidates(llm, m, cfg) for m in todo]), 1):
        module, answers = await finished
        # On garde le bloc de code de chaque réponse ; une réponse coupée ou sans code est jetée.
        tests = [code for code in (extract_code(a) for a in answers) if code]
        if not tests:
            # Rien n'est écrit : le module sera retenté à la prochaine relance.
            print(f"  [{k}/{len(todo)}] {module['id']} : aucun candidat exploitable")
            continue
        append_jsonl(cfg["paths"]["teacher_gen"], {
            "id": module["id"], "lang": module["lang"], "code": module["code"],
            "teacher": llm.model, "candidates": tests,
        })
        print(f"  [{k}/{len(todo)}] {module['id']} : {len(tests)} candidats")

    if llm.providers:
        print(f"Fournisseurs : {dict(llm.providers)}")


if __name__ == "__main__":
    asyncio.run(main())
