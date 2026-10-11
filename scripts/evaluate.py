"""Évaluation : chaque système (base, TestMaker, teacher) écrit des tests pour les modules du split test,
puis ces tests sont jugés exactement comme ceux du teacher à la validation (sandbox + mutants + élagage).

    python scripts/evaluate.py                                # tous les systèmes, tous les modules test
    python scripts/evaluate.py --systems testmaker base --limit 5
    python scripts/evaluate.py --run v2                       # autre dossier de résultats (ex. après réentraînement)
    python scripts/evaluate.py --report                       # réaffiche le bilan sans rien générer

Résultats dans data/eval/<run>/ : results.jsonl (une ligne par module et par système, reprise
automatique en cas d'arrêt) et report.md (le tableau comparatif).

Les systèmes sont définis dans config.yaml (`eval.systems`). Pour générer sur un GPU RunPod au lieu
du CPU du Mac, servir les modèles avec vLLM sur le pod et remplacer leur `base_url` par celle du pod.
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

# Une fonction de test pytest : `def test_x(` ou `async def test_x(`, éventuellement dans une classe.
_TEST_FUNC = re.compile(r"^\s*(?:async\s+)?def test_", re.M)


def make_llm(cfg: dict, profile: dict) -> LLM:
    """Client d'un système : soit un profil de teacher existant, soit un endpoint décrit en entier."""
    if "teacher" in profile:
        return teacher_llm(cfg, profile["teacher"])
    return LLM(profile["model"], profile["base_url"], profile["api_key_env"],
               profile.get("concurrency", 1), profile.get("timeout_s", 300))


async def run_system(name: str, llm: LLM, todo: list[dict], cfg: dict, out: Path,
                     pool: ThreadPoolExecutor) -> None:
    """Génère puis juge les tests d'un système pour chaque module de `todo`."""
    ev, loop = cfg["eval"], asyncio.get_running_loop()

    async def one(rec: dict) -> dict:
        start = time.time()
        try:
            text, error = await llm.complete(build_messages(rec["lang"], rec["code"]),
                                             ev["temperature"], ev["max_tokens"]), None
        except Exception as e:  # une génération perdue compte comme un échec, sans arrêter l'évaluation
            text, error = "", f"{type(e).__name__}: {e}"
        row = {"id": rec["id"], "system": name, "model": llm.model,
               "gen_s": round(time.time() - start, 1), "error": error}
        test = extract_code(text)
        if not test:
            return row | {"source_ok": True, "raw_passed": False, "passed": False, "pruned": 0,
                          "mutation_score": None, "n_tests_raw": 0, "n_tests": 0, "test": text}
        muts = mutants(rec["code"], rec["lang"], cfg["sandbox"]["n_mutants"], seed=rec["id"])
        # La sandbox est bloquante (Docker) : elle tourne dans un thread pour ne pas figer les générations.
        result, final = await loop.run_in_executor(pool, check_test, rec["lang"], rec["code"], test, muts, cfg)
        row |= {k: result[k] for k in ("source_ok", "raw_passed", "passed", "pruned",
                                       "mutants_total", "mutants_killed", "mutation_score")}
        row |= {"n_tests_raw": len(_TEST_FUNC.findall(test)), "n_tests": len(_TEST_FUNC.findall(final)),
                "test": test, "final_test": final}
        if not result["passed"]:
            row["log"] = result["log"][-1500:]
        return row

    for k, fut in enumerate(asyncio.as_completed([one(r) for r in todo]), 1):
        row = await fut
        append_jsonl(out, row)   # écrit au fil de l'eau : un arrêt ne perd que les modules en cours
        verdict = (f"erreur ({row['error'][:60]})" if row["error"] else
                   "pas de code" if not row["n_tests_raw"] else
                   ("passe" if row["raw_passed"] else f"passe après élagage (-{row['pruned']})" if row["passed"]
                    else "échoue") + (f", mutants {row['mutants_killed']}/{row['mutants_total']}"
                                      if row["passed"] else ""))
        print(f"  [{name} {k}/{len(todo)}] {row['id']} : {verdict}  ({row['gen_s']} s)")


def _pct(values: list[bool]) -> str:
    return f"{100 * sum(values) / len(values):.0f} %" if values else "-"


def _mean(values: list[float], fmt: str = "{:.2f}") -> str:
    return fmt.format(statistics.mean(values)) if values else "-"


def report(rows: list[dict], cfg: dict, out_md: Path) -> None:
    """Tableau comparatif, calculé sur les modules évalués par tous les systèmes présents."""
    threshold = cfg["filter"]["min_mutation_score"]
    systems = list(dict.fromkeys(r["system"] for r in rows))
    common = set.intersection(*({r["id"] for r in rows if r["system"] == s} for s in systems))
    lines = [f"Évaluation sur {len(common)} modules du split test (communs à : {', '.join(systems)}).", "",
             "| Système | Passe tel quel | Passe après élagage | Tests utiles | Score de mutation | "
             "Tests par fichier | Génération |",
             "|---|---|---|---|---|---|---|"]
    for s in systems:
        rs = [r for r in rows if r["system"] == s and r["id"] in common]
        ok = [r for r in rs if r["passed"]]
        scores = [r["mutation_score"] for r in ok if r["mutation_score"] is not None]
        # « Utile » = même règle que pour garder un exemple du teacher : passe et détecte assez de mutants.
        useful = [r["passed"] and (r["mutation_score"] is None or r["mutation_score"] >= threshold) for r in rs]
        lines.append(f"| {s} ({rs[0]['model']}) | {_pct([r['raw_passed'] for r in rs])} | "
                     f"{_pct([r['passed'] for r in rs])} | {_pct(useful)} | {_mean(scores)} | "
                     f"{_mean([r['n_tests'] for r in ok], '{:.0f}')} | {_mean([r['gen_s'] for r in rs], '{:.0f} s')} |")
    lines += ["",
              "- **Passe tel quel** : le fichier généré passe sur le code original sans retouche.",
              "- **Passe après élagage** : au plus 30 % de fonctions de test fausses, retirées automatiquement.",
              f"- **Tests utiles** : passe (après élagage) et tue au moins {threshold:.0%} des mutants.",
              "- **Score de mutation** : part des bugs introduits détectés, sur les fichiers qui passent.",
              "- **Tests par fichier** : fonctions de test restantes, sur les fichiers qui passent."]
    text = "\n".join(lines) + "\n"
    out_md.write_text(text)
    print("\n" + text + f"\nRapport : {out_md}")


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
    out = run_dir / "results.jsonl"
    if not args.report:
        modules = read_jsonl(Path(cfg["paths"]["sft_dir"]) / "test.jsonl")[: args.limit]
        done = {(r["system"], r["id"]) for r in read_jsonl(out)}
        with ThreadPoolExecutor(cfg["sandbox"]["workers"]) as pool:
            for name in args.systems:
                llm = make_llm(cfg, cfg["eval"]["systems"][name])
                todo = [m for m in modules if (name, m["id"]) not in done]
                if not todo:
                    continue
                try:  # vérifie que le modèle répond avant de lancer 25 générations (modèle absent, clé manquante...)
                    await llm.complete([{"role": "user", "content": "ok"}], 0, 1)
                except Exception as e:
                    print(f"{name} ({llm.model}) ignoré : {type(e).__name__}: {e}")
                    continue
                print(f"\n{name} ({llm.model}) : {len(todo)} modules")
                await run_system(name, llm, todo, cfg, out, pool)
    rows = [r for r in read_jsonl(out) if r["system"] in args.systems]
    if rows:
        report(rows, cfg, run_dir / "report.md")


if __name__ == "__main__":
    asyncio.run(main())
