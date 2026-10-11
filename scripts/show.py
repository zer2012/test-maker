"""Affiche tout ce que le pipeline sait d'une source, lisiblement, dans un fichier Markdown.

    python scripts/show.py                      # liste les ids et leur état
    python scripts/show.py synth-python-00009   # écrit data/view/synth-python-00009.md
    python scripts/show.py 9                    # raccourci : l'id qui se termine par 00009

Ouvrir ensuite le .md dans VS Code avec l'aperçu Markdown (Cmd+Shift+V) : le code apparaît
avec de vrais retours à la ligne et la coloration syntaxique.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import load_config, read_jsonl, split_of
from testmaker.languages import LANGUAGES


def _by_id(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(path)}


def _status(val: dict | None) -> str:
    if val is None:
        return "pas encore validé"
    if val["best"] is None:
        return "rejeté (aucun candidat ne passe)"
    r = val["results"][val["best"]]
    return f"best={val['best']} mutants {r['mutants_killed']}/{r['mutants_total']}"


def main() -> None:
    cfg = load_config()
    p = cfg["paths"]
    sources, gen, val = _by_id(p["sources"]), _by_id(p["teacher_gen"]), _by_id(p["teacher_val"])

    if len(sys.argv) < 2:
        for rid in sources:
            print(f"{rid}  [{split_of(rid):5}]  {_status(val.get(rid))}")
        return

    arg = sys.argv[1]
    # Accepte l'id complet ou juste son numéro (« 9 » -> « ...-00009 »).
    matches = [rid for rid in sources if rid == arg or (arg.isdigit() and rid.endswith(f"-{int(arg):05d}"))]
    if not matches:
        sys.exit(f"id inconnu : {arg}")
    rid = matches[0]
    src, g, v = sources[rid], gen.get(rid), val.get(rid)
    fence = LANGUAGES[src["lang"]].fence

    md = [f"# {rid}", "", f"- split : **{split_of(rid)}**", f"- état : **{_status(v)}**", "",
          "## Code (étape 1)", "", f"```{fence}", src["code"], "```", ""]
    if g:
        # `tests` = versions élaguées (étape 3) ; sinon les candidats bruts du teacher (étape 2).
        tests = v.get("tests", g["candidates"]) if v else g["candidates"]
        for i, test in enumerate(tests):
            title = f"## Candidat {i}"
            if v:
                r = v["results"][i]
                verdict = ("passe" if r["passed"] else "échoue") + (
                    f", mutants {r['mutants_killed']}/{r['mutants_total']}" if r["passed"] else "")
                if r.get("pruned"):
                    verdict += f", {r['pruned']} test(s) retiré(s)"
                title += f" ({verdict})" + ("  ⭐ meilleur" if v["best"] == i else "")
            md += [title, "", f"```{fence}", test, "```", ""]
            if v and not v["results"][i]["passed"]:
                md += ["<details><summary>log</summary>", "", "```", v["results"][i]["log"], "```",
                       "", "</details>", ""]

    out = Path(p["sources"]).parent / "view" / f"{rid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
