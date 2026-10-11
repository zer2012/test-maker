"""Évaluation : chaque système (base, TestMaker, teacher) écrit des tests pour les modules du split test,
puis ces tests sont jugés comme ceux du teacher à la validation (sandbox + mutants + élagage).

    python scripts/evaluate.py                                # tous les systèmes, tous les modules test
    python scripts/evaluate.py --systems testmaker base --limit 5
    python scripts/evaluate.py --run v2                       # autre dossier de résultats (ex. après réentraînement)
    python scripts/evaluate.py --report                       # réaffiche le bilan sans rien générer

Résultats dans data/eval/<run>/ :
    results.jsonl  une ligne par (système, module) ; une relance saute ce qui est déjà fait
    report.md      le tableau comparatif

Les systèmes sont décrits dans config.yaml, section `eval.systems`. Un système = un modèle appelé
par son NOM à une adresse (ex. le modèle « testmaker » d'Ollama sur localhost:11434).
"""

import argparse
import asyncio
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.check import check_test
from testmaker.common import append_jsonl, extract_code, load_config, read_jsonl
from testmaker.llm import LLM, teacher_llm
from testmaker.mutate import mutants
from testmaker.prompts import build_messages

# Repère une fonction de test pytest (`def test_...`), pour compter les tests d'un fichier.
TEST_FUNCTION = re.compile(r"^\s*(?:async\s+)?def test_", re.M)


# --- Préparer les systèmes ------------------------------------------------------------------

def make_llm(cfg: dict, system: dict) -> LLM:
    """Crée le client d'un système décrit dans config.yaml (eval.systems.<nom>)."""
    if "teacher" in system:
        # `teacher: openrouter` : on réutilise le profil déjà défini pour la génération des données.
        return teacher_llm(cfg, system["teacher"])
    return LLM(
        model=system["model"],                # nom du modèle chez le serveur, ex. "testmaker"
        base_url=system["base_url"],          # adresse du serveur, ex. Ollama en local
        api_key_env=system["api_key_env"],
        concurrency=system.get("concurrency", 1),
        timeout=system.get("timeout_s", 300),
    )


async def responds(llm: LLM) -> bool:
    """Petit appel d'essai : évite de lancer 25 générations sur un modèle absent ou une clé manquante."""
    try:
        await llm.complete([{"role": "user", "content": "ok"}], temperature=0, max_tokens=1)
        return True
    except Exception as e:
        print(f"{llm.model} ne répond pas, système ignoré : {type(e).__name__}: {e}")
        return False


# --- Évaluer un module ------------------------------------------------------------------------

async def generate(llm: LLM, module: dict, cfg: dict) -> tuple[str, float, str | None]:
    """Demande des tests au modèle. Renvoie (réponse, durée en secondes, erreur éventuelle)."""
    start = time.time()
    try:
        answer = await llm.complete(build_messages(module["lang"], module["code"]),
                                    temperature=cfg["eval"]["temperature"],
                                    max_tokens=cfg["eval"]["max_tokens"])
        error = None
    except Exception as e:
        # Une génération perdue compte comme un échec, sans arrêter toute l'évaluation.
        answer, error = "", f"{type(e).__name__}: {e}"
    return answer, round(time.time() - start, 1), error


def judge(module: dict, test: str, cfg: dict) -> dict:
    """Exécute le test dans la sandbox (original + mutants, élagage si besoin) et résume le verdict."""
    muts = mutants(module["code"], module["lang"], cfg["sandbox"]["n_mutants"], seed=module["id"])
    result, final_test = check_test(module["lang"], module["code"], test, muts, cfg)
    verdict = {
        "raw_passed": result["raw_passed"],          # passe tel quel
        "passed": result["passed"],                  # passe, éventuellement après élagage
        "pruned": result["pruned"],                  # fonctions de test retirées
        "mutants_killed": result["mutants_killed"],
        "mutants_total": result["mutants_total"],
        "mutation_score": result["mutation_score"],
        "n_tests_raw": len(TEST_FUNCTION.findall(test)),
        "n_tests": len(TEST_FUNCTION.findall(final_test)),
        "final_test": final_test,
    }
    if not result["passed"]:
        verdict["log"] = result["log"][-1500:]       # fin du log pytest, pour comprendre l'échec
    return verdict


def no_code_verdict() -> dict:
    """Verdict quand la réponse ne contient aucun bloc de code : tout est compté comme échoué."""
    return {"raw_passed": False, "passed": False, "pruned": 0, "mutants_killed": 0, "mutants_total": 0,
            "mutation_score": None, "n_tests_raw": 0, "n_tests": 0}


def describe(row: dict) -> str:
    """Une ligne lisible pour le suivi dans le terminal."""
    if row["error"]:
        return f"erreur ({row['error'][:60]})"
    if row["n_tests_raw"] == 0:
        return "pas de code dans la réponse"
    if not row["passed"]:
        return "échoue"
    status = "passe" if row["raw_passed"] else f"passe après élagage (-{row['pruned']})"
    return f"{status}, mutants {row['mutants_killed']}/{row['mutants_total']}"


async def evaluate_system(name: str, llm: LLM, modules: list[dict], cfg: dict, out: Path,
                          pool: ThreadPoolExecutor) -> None:
    """Génère et juge les tests d'un système pour chaque module, en enregistrant au fil de l'eau."""
    loop = asyncio.get_running_loop()

    async def evaluate_module(module: dict) -> dict:
        answer, seconds, error = await generate(llm, module, cfg)
        test = extract_code(answer)
        if test:
            # La sandbox bloque (elle attend Docker) : on la lance dans un thread à part.
            verdict = await loop.run_in_executor(pool, judge, module, test, cfg)
        else:
            verdict = no_code_verdict()
        return {"id": module["id"], "system": name, "model": llm.model, "gen_s": seconds,
                "error": error, "test": test or answer, **verdict}

    tasks = [evaluate_module(m) for m in modules]
    for k, finished in enumerate(asyncio.as_completed(tasks), 1):
        row = await finished
        append_jsonl(out, row)
        print(f"  [{name} {k}/{len(modules)}] {row['id']} : {describe(row)}  ({row['gen_s']} s)")


# --- Bilan --------------------------------------------------------------------------------------

def percent(flags: list[bool]) -> str:
    return f"{100 * sum(flags) / len(flags):.0f} %" if flags else "-"


def average(values: list[float], fmt: str) -> str:
    return fmt.format(statistics.mean(values)) if values else "-"


def system_line(name: str, rows: list[dict], threshold: float) -> str:
    """Une ligne du tableau pour un système."""
    passing = [r for r in rows if r["passed"]]
    scores = [r["mutation_score"] for r in passing if r["mutation_score"] is not None]
    # « Utile » = même règle que pour garder un exemple du teacher : passe et détecte assez de mutants.
    useful = [r["passed"] and (r["mutation_score"] is None or r["mutation_score"] >= threshold) for r in rows]
    cells = [
        f"{name} ({rows[0]['model']})",
        percent([r["raw_passed"] for r in rows]),
        percent([r["passed"] for r in rows]),
        percent(useful),
        average(scores, "{:.2f}"),
        average([r["n_tests"] for r in passing], "{:.0f}"),
        average([r["gen_s"] for r in rows], "{:.0f} s"),
    ]
    return "| " + " | ".join(cells) + " |"


def report(rows: list[dict], cfg: dict, out_md: Path) -> None:
    """Tableau comparatif, calculé uniquement sur les modules évalués par TOUS les systèmes présents."""
    threshold = cfg["filter"]["min_mutation_score"]
    systems = list(dict.fromkeys(r["system"] for r in rows))
    ids_per_system = [{r["id"] for r in rows if r["system"] == s} for s in systems]
    common = set.intersection(*ids_per_system)

    lines = [
        f"Évaluation sur {len(common)} modules du split test (communs à : {', '.join(systems)}).",
        "",
        "| Système | Passe tel quel | Passe après élagage | Tests utiles | Score de mutation "
        "| Tests par fichier | Génération |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in systems:
        lines.append(system_line(s, [r for r in rows if r["system"] == s and r["id"] in common], threshold))
    lines += [
        "",
        "- **Passe tel quel** : le fichier généré passe sur le code original sans retouche.",
        "- **Passe après élagage** : au plus 30 % de fonctions de test fausses, retirées automatiquement.",
        f"- **Tests utiles** : passe (après élagage) et tue au moins {threshold:.0%} des mutants.",
        "- **Score de mutation** : part des bugs introduits détectés, sur les fichiers qui passent.",
        "- **Tests par fichier** : fonctions de test restantes, sur les fichiers qui passent.",
    ]
    text = "\n".join(lines) + "\n"
    out_md.write_text(text)
    print("\n" + text + f"\nRapport : {out_md}")


# --- Programme principal ----------------------------------------------------------------------

async def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(cfg["eval"]["systems"]))
    ap.add_argument("--limit", type=int, help="ne prendre que les N premiers modules du split test")
    ap.add_argument("--run", default="v1", help="nom du dossier de résultats dans data/eval/")
    ap.add_argument("--report", action="store_true", help="réaffiche le bilan sans rien générer")
    args = ap.parse_args()

    run_dir = Path(cfg["paths"]["sources"]).parent / "eval" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    results_file = run_dir / "results.jsonl"

    if not args.report:
        modules = read_jsonl(Path(cfg["paths"]["sft_dir"]) / "test.jsonl")[: args.limit]
        already_done = {(r["system"], r["id"]) for r in read_jsonl(results_file)}
        with ThreadPoolExecutor(cfg["sandbox"]["workers"]) as pool:
            for name in args.systems:
                todo = [m for m in modules if (name, m["id"]) not in already_done]
                if not todo:
                    continue
                llm = make_llm(cfg, cfg["eval"]["systems"][name])
                if not await responds(llm):
                    continue
                print(f"\n{name} ({llm.model}) : {len(todo)} modules")
                await evaluate_system(name, llm, todo, cfg, results_file, pool)

    rows = [r for r in read_jsonl(results_file) if r["system"] in args.systems]
    if rows:
        report(rows, cfg, run_dir / "report.md")


if __name__ == "__main__":
    asyncio.run(main())
