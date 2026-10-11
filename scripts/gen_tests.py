"""Étape 2 : le teacher écrit plusieurs candidats de tests par source -> data/teacher_gen.jsonl.

    python scripts/gen_tests.py --lang python [--limit 10]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import append_jsonl, extract_code, load_config, read_jsonl
from testmaker.llm import teacher_llm
from testmaker.prompts import build_messages


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--provider")
    args = ap.parse_args()

    cfg = load_config()
    t = cfg["teacher"]
    src, out = cfg["paths"]["sources"], cfg["paths"]["teacher_gen"]
    llm = teacher_llm(cfg, args.provider)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(src) if r["id"] not in done and (not args.lang or r["lang"] == args.lang)]
    todo = todo[: args.limit]
    print(f"{len(todo)} sources x {t['n_candidates']} candidats avec {llm.model}")

    async def one(rec: dict) -> tuple[dict, list[str]]:
        """n_candidates réponses du teacher pour une source."""
        texts = await llm.sample(build_messages(rec["lang"], rec["code"]), n=t["n_candidates"],
                                 temperature=t["temperature"], max_tokens=t["max_tokens"])
        return rec, texts

    for k, fut in enumerate(asyncio.as_completed([one(r) for r in todo]), 1):
        rec, texts = await fut
        # Garde le bloc de code de chaque réponse ; les réponses sans code exploitable sont jetées.
        tests = [c for c in (extract_code(x) for x in texts) if c]
        if not tests:
            print(f"  [{k}/{len(todo)}] {rec['id']} : aucun candidat exploitable")
            continue
        append_jsonl(out, {"id": rec["id"], "lang": rec["lang"], "code": rec["code"],
                           "teacher": llm.model, "candidates": tests})
        print(f"  [{k}/{len(todo)}] {rec['id']} : {len(tests)} candidats")


if __name__ == "__main__":
    asyncio.run(main())
