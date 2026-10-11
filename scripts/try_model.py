"""Fait écrire des tests par un modèle servi par Ollama, puis les exécute dans la sandbox.

    python scripts/try_model.py                          # TestMaker sur le 1er module du split test
    python scripts/try_model.py 3                        # 4e module du split test
    python scripts/try_model.py --file chemin/module.py  # n'importe quel fichier Python
    python scripts/try_model.py --model qwen2.5-coder:7b # un autre modèle Ollama, pour comparer

Le test généré est enregistré dans data/tries/<modèle>/<id>_test.py.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import extract_code, load_config, read_jsonl
from testmaker.llm import LLM
from testmaker.mutate import mutants
from testmaker.prompts import build_messages
from testmaker.sandbox import run_tests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, default=0, help="numéro de ligne dans data/sft/test.jsonl")
    ap.add_argument("--file", help="fichier Python à tester à la place d'un module du split test")
    ap.add_argument("--model", default="testmaker")
    args = ap.parse_args()

    cfg = load_config()
    if args.file:
        rid, code = Path(args.file).stem, Path(args.file).read_text()
    else:
        rec = read_jsonl(Path(cfg["paths"]["sft_dir"]) / "test.jsonl")[args.index]
        rid, code = rec["id"], rec["code"]
    print(f"{rid} : {len(code.splitlines())} lignes, modèle {args.model}")

    # Même prompt qu'à l'entraînement ; température 0 = la réponse la plus probable, reproductible.
    llm = LLM(args.model, "http://localhost:11434/v1", "OLLAMA_API_KEY", concurrency=1, timeout=1800)
    start = time.time()
    text = asyncio.run(llm.complete(build_messages("python", code), temperature=0, max_tokens=3072))
    test = extract_code(text)
    print(f"Généré en {time.time() - start:.0f} s ({len(test.splitlines())} lignes de test)")
    if not test:
        print("Aucun bloc de code dans la réponse :\n" + text[:2000])
        return

    out = Path(cfg["paths"]["sources"]).parent / "tries" / args.model.replace(":", "_") / f"{rid}_test.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(test)
    print(f"Test enregistré : {out}")

    # Validation identique à celle des tests du teacher : code original, puis 3 mutants.
    r = run_tests("python", code, test, mutants(code, "python", cfg["sandbox"]["n_mutants"], seed=rid), cfg)
    print(json.dumps({"passe": r.passed, "mutants_tués": f"{r.mutants_killed}/{r.mutants_total}"}, ensure_ascii=False))
    if not r.passed:
        print("\n--- fin du log pytest ---\n" + r.log[-2000:])


if __name__ == "__main__":
    main()
