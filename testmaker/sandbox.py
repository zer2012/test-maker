"""Exécution isolée des tests générés dans Docker (sans réseau, CPU/RAM limités).

Un seul conteneur par (code, test) : on exécute d'abord le code original, puis, si les
tests passent, chaque mutant. Un mutant qui ne compile pas est ignoré.

Disposition du dossier temporaire monté dans le conteneur sous /work :
    run.sh            script shell généré par `_script`
    orig/             code original + fichier de test
    m0/, m1/, ...     un mutant chacun, avec le même fichier de test
    orig.log, m0.log  sortie des tests ; *.build.log : sortie de la compilation
"""

import re
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .languages import LANGUAGES

BUILD_FAIL = "BUILD_FAIL"


@dataclass
class RunResult:
    source_ok: bool          # le code original compile
    passed: bool             # les tests passent sur le code original
    mutants_total: int       # mutants qui compilent
    mutants_killed: int      # mutants détectés par les tests
    log: str                 # fin de la sortie du run original

    @property
    def mutation_score(self) -> float | None:
        """Part des mutants détectés ; None si aucun mutant n'a pu être évalué."""
        return self.mutants_killed / self.mutants_total if self.mutants_total else None

    def to_dict(self) -> dict:
        """Version sérialisable en JSON, score de mutation inclus."""
        return asdict(self) | {"mutation_score": self.mutation_score}


def _script(lang: str, code: str, variants: list[str], timeout: int) -> str:
    """Génère le script shell exécuté dans le conteneur.

    Chaque variante (orig, m0, m1...) est compilée puis testée ; le script affiche une ligne
    `RESULT <variante> <code de sortie | BUILD_FAIL>` que `run_tests` relit ensuite.
    """
    spec = LANGUAGES[lang]
    # shlex.quote protège les commandes (espaces, &&, *) pour les passer à `sh -c '...'`.
    build = shlex.quote(spec.build_cmd.format(code_file=spec.code_file(code)))
    test = shlex.quote(spec.test_cmd)
    lines = [
        "run() {",  # fonction shell ; $1 = nom du dossier de la variante
        # 1) compilation, limitée par `timeout` ; en cas d'échec on s'arrête là pour cette variante.
        f'  if ! (cd "$1" && timeout {timeout} sh -c {build}) > "$1.build.log" 2>&1; then',
        f'    echo "RESULT $1 {BUILD_FAIL}"; return; fi',
        # 2) tests ; leur code de sortie ($?) vaut 0 si tout passe, autre chose sinon (124 = timeout).
        f'  (cd "$1" && timeout {timeout} sh -c {test}) > "$1.log" 2>&1',
        '  echo "RESULT $1 $?"',
        "}",
        # Original d'abord : si les tests n'y passent pas, inutile d'essayer les mutants.
        'r=$(run orig); echo "$r"; [ "$r" = "RESULT orig 0" ] || exit 0',
    ]
    lines += [f"run {v}" for v in variants if v != "orig"]
    return "\n".join(lines) + "\n"


def run_tests(lang: str, code: str, test: str, mutant_codes: list[str], cfg: dict) -> RunResult:
    """Exécute `test` contre `code` puis contre chaque mutant, dans un conteneur jetable."""
    spec, sb = LANGUAGES[lang], cfg["sandbox"]
    # Dossier temporaire dans le projet (pas /tmp) : Docker Desktop ne peut monter que les
    # dossiers partagés avec sa VM, et /Users l'est par défaut.
    tmp_root = Path(cfg["paths"]["sandbox_tmp"]).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_root, ignore_cleanup_errors=True) as tmp:
        # Écrit un sous-dossier complet (code + test + fichiers de projet) par variante.
        variants = {"orig": code} | {f"m{i}": m for i, m in enumerate(mutant_codes)}
        for name, variant_code in variants.items():
            for rel, content in spec.files(variant_code, test).items():
                path = Path(tmp, name, rel)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        Path(tmp, "run.sh").write_text(_script(lang, code, list(variants), sb["timeout_s"]))

        # --rm : conteneur supprimé à la fin ; --network none : le code testé n'a pas d'accès réseau ;
        # --memory / --cpus / --pids-limit : bornent la RAM, le CPU et le nombre de processus
        # (protège contre les boucles infinies et les fork bombs) ; -v monte le dossier en /work.
        cmd = ["docker", "run", "--rm", "--network", "none", f"--memory={sb['memory']}",
               f"--cpus={sb['cpus']}", "--pids-limit", "256", "-v", f"{tmp}:/work", "-w", "/work",
               spec.image, "sh", "run.sh"]
        try:
            # Garde-fou global au cas où `timeout` ne suffirait pas : build + test par variante, plus marge.
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=2 * sb["timeout_s"] * len(variants) + 60)
            stdout = proc.stdout
        except subprocess.TimeoutExpired:
            stdout = ""

        # {"orig": "0", "m0": "1", "m1": "BUILD_FAIL", ...}
        results = dict(re.findall(r"^RESULT (\S+) (\S+)$", stdout, re.M))
        # On conserve la fin du log de l'original (tests, ou compilation si elle a échoué) : pour le debug,
        # et pour l'élagage qui y lit le résumé `FAILED ...` de pytest.
        log_file = Path(tmp, "orig.log" if results.get("orig") != BUILD_FAIL else "orig.build.log")
        log = log_file.read_text(errors="replace")[-4000:] if log_file.exists() else stdout[-4000:]

    orig = results.get("orig")
    # Codes de sortie des mutants qui ont compilé ; un code non nul = au moins un test a échoué = mutant tué.
    mutant_exit_codes = [v for k, v in results.items() if k != "orig" and v != BUILD_FAIL]
    return RunResult(
        source_ok=orig is not None and orig != BUILD_FAIL,
        passed=orig == "0",
        mutants_total=len(mutant_exit_codes),
        mutants_killed=sum(v != "0" for v in mutant_exit_codes),
        log=log,
    )
