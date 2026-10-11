"""Affiche tout ce que le pipeline sait d'un module, lisiblement, dans une page Markdown.

    python scripts/show.py                      # liste les modules, leur split et leur état
    python scripts/show.py synth-python-00009   # écrit data/view/synth-python-00009.md
    python scripts/show.py 9                    # raccourci : le module dont l'id se termine par 00009

La page contient le code du module, les candidats du teacher avec leur verdict, et, pour un module
du split test, les tests écrits par chaque modèle évalué (data/eval/<run>/results.jsonl).
L'ouvrir dans VS Code avec l'aperçu Markdown (Cmd+Shift+V) : vrais retours à la ligne et couleurs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import load_config, read_jsonl, split_of
from testmaker.languages import LANGUAGES


# --- Petits outils de mise en page ----------------------------------------------------------------

def code_block(code: str, fence: str = "") -> list[str]:
    return [f"```{fence}", code, "```", ""]


def foldable(title: str, content: str, fence: str = "") -> list[str]:
    """Section repliée par défaut dans l'aperçu Markdown (pour les logs et les versions brutes)."""
    return [f"<details><summary>{title}</summary>", "", *code_block(content, fence), "</details>", ""]


def by_id(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


# --- Verdicts lisibles ------------------------------------------------------------------------

def validation_status(record: dict | None) -> str:
    """État d'un module après la validation (data/teacher_val.jsonl)."""
    if record is None:
        return "pas encore validé"
    if record["best"] is None:
        return "rejeté (aucun candidat ne passe)"
    r = record["results"][record["best"]]
    return f"best={record['best']} mutants {r['mutants_killed']}/{r['mutants_total']}"


def candidate_verdict(r: dict) -> str:
    """Verdict d'un candidat du teacher."""
    if not r["passed"]:
        return "échoue"
    verdict = f"passe, mutants {r['mutants_killed']}/{r['mutants_total']}"
    return verdict + (f", {r['pruned']} test(s) retiré(s)" if r.get("pruned") else "")


def eval_verdict(r: dict) -> str:
    """Verdict d'une ligne d'évaluation (data/eval/<run>/results.jsonl)."""
    if r.get("error"):
        return f"erreur : {r['error'][:80]}"
    if not r.get("n_tests_raw"):
        return "pas de code (réponse coupée ou sans bloc de code)"
    if not r["passed"]:
        return f"échoue ({r['n_tests_raw']} fonctions de test)"
    status = "passe tel quel" if r["raw_passed"] else f"passe après élagage ({r['pruned']} retirée(s))"
    return f"{status}, mutants {r['mutants_killed']}/{r['mutants_total']}"


# --- Les sections de la page ------------------------------------------------------------------

def teacher_section(gen: dict | None, val: dict | None, fence: str) -> list[str]:
    """Les candidats du teacher : versions élaguées et verdicts si le module est validé, sinon bruts."""
    if gen is None:
        return []
    tests = val.get("tests", gen["candidates"]) if val else gen["candidates"]
    md = []
    for i, test in enumerate(tests):
        title = f"## Candidat {i}"
        if val:
            title += f" ({candidate_verdict(val['results'][i])})"
            title += "  ⭐ meilleur" if val["best"] == i else ""
        md += [title, "", *code_block(test, fence)]
        if val and not val["results"][i]["passed"]:
            md += foldable("log", val["results"][i]["log"])
    return md


def evaluation_section(rid: str, data_dir: Path, fence: str) -> list[str]:
    """Les tests écrits pour ce module par chaque modèle évalué, run par run."""
    md = []
    for results in sorted((data_dir / "eval").glob("*/results.jsonl")):
        for r in read_jsonl(results):
            if r["id"] != rid:
                continue
            md += [f"## Évaluation {results.parent.name} — {r['system']} ({r['model']})", "",
                   f"**{eval_verdict(r)}** · génération {r['gen_s']} s", ""]
            if r.get("n_tests_raw"):
                md += code_block(r.get("final_test") or r["test"], fence)
                if r.get("pruned"):
                    md += foldable("version brute, avant élagage", r["test"], fence)
            else:
                md += foldable("réponse brute (sans bloc de code complet)", r["test"])
            if r.get("log"):
                md += foldable("log pytest", r["log"])
    return md


# --- Programme principal ----------------------------------------------------------------------

def find_id(arg: str, ids) -> str:
    """Accepte l'id complet ou juste son numéro (« 9 » -> « synth-python-00009 »)."""
    matches = [rid for rid in ids if rid == arg or (arg.isdigit() and rid.endswith(f"-{int(arg):05d}"))]
    if not matches:
        sys.exit(f"id inconnu : {arg}")
    return matches[0]


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    data_dir = Path(paths["sources"]).parent
    sources, gen, val = by_id(paths["sources"]), by_id(paths["teacher_gen"]), by_id(paths["teacher_val"])

    if len(sys.argv) < 2:   # sans argument : la liste des modules
        for rid in sources:
            print(f"{rid}  [{split_of(rid):5}]  {validation_status(val.get(rid))}")
        return

    rid = find_id(sys.argv[1], sources)
    fence = LANGUAGES[sources[rid]["lang"]].fence
    md = [f"# {rid}", "",
          f"- split : **{split_of(rid)}**",
          f"- état : **{validation_status(val.get(rid))}**", "",
          "## Code (étape 1)", "", *code_block(sources[rid]["code"], fence)]
    md += teacher_section(gen.get(rid), val.get(rid), fence)
    md += evaluation_section(rid, data_dir, fence)

    out = data_dir / "view" / f"{rid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
