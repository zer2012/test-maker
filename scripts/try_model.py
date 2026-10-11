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

OLLAMA_URL = "http://localhost:11434/v1"


def load_module(args, cfg: dict) -> tuple[str, str]:
    """Le code à tester : un fichier donné avec --file, sinon un module du split test. Renvoie (id, code)."""
    if args.file:
        return Path(args.file).stem, Path(args.file).read_text()
    module = read_jsonl(Path(cfg["paths"]["sft_dir"]) / "test.jsonl")[args.index]
    return module["id"], module["code"]


def generate_test(model: str, code: str) -> str:
    """Demande des tests au modèle avec le prompt d'entraînement. Renvoie la réponse brute."""
    llm = LLM(model, OLLAMA_URL, "OLLAMA_API_KEY", concurrency=1, timeout=1800)
    # Température 0 : la réponse la plus probable, reproductible d'un essai à l'autre.
    return asyncio.run(llm.complete(build_messages("python", code), temperature=0, max_tokens=3072))


def save_test(test: str, model: str, rid: str, cfg: dict) -> Path:
    # Les « : » des noms de modèles Ollama (qwen2.5-coder:7b) ne sont pas pratiques dans un nom de dossier.
    out = Path(cfg["paths"]["sources"]).parent / "tries" / model.replace(":", "_") / f"{rid}_test.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(test)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, default=0, help="numéro de ligne dans data/sft/test.jsonl")
    ap.add_argument("--file", help="fichier Python à tester à la place d'un module du split test")
    ap.add_argument("--model", default="testmaker", help="nom du modèle dans Ollama")
    args = ap.parse_args()

    cfg = load_config()
    rid, code = load_module(args, cfg)
    print(f"{rid} : {len(code.splitlines())} lignes, modèle {args.model}")

    start = time.time()
    answer = generate_test(args.model, code)
    test = extract_code(answer)
    print(f"Généré en {time.time() - start:.0f} s ({len(test.splitlines())} lignes de test)")
    if not test:
        print("Aucun bloc de code dans la réponse :\n" + answer[:2000])
        return
    print(f"Test enregistré : {save_test(test, args.model, rid, cfg)}")

    # Même jugement que pour les tests du teacher : code original, puis les mutants.
    muts = mutants(code, "python", cfg["sandbox"]["n_mutants"], seed=rid)
    result = run_tests("python", code, test, muts, cfg)
    print(json.dumps({"passe": result.passed, "mutants_tués": f"{result.mutants_killed}/{result.mutants_total}"},
                     ensure_ascii=False))
    if not result.passed:
        print("\n--- fin du log pytest ---\n" + result.log[-2000:])


if __name__ == "__main__":
    main()
