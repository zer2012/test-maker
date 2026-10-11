"""Vérifie la sandbox sans LLM : un bon test, un test creux et un test faux sur un même code.

Attendu : le bon test passe et tue des mutants, le test creux passe mais ne tue rien,
le test faux échoue.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testmaker.common import load_config
from testmaker.mutate import mutants
from testmaker.sandbox import run_tests

EX = Path(__file__).resolve().parent.parent / "examples" / "python"


def main() -> int:
    cfg = load_config()
    code = (EX / "solution.py").read_text()
    muts = mutants(code, "python", 6, seed="selftest")
    print(f"{len(muts)} mutants générés")

    expectations = {
        "test_good.py": lambda r: r.passed and r.mutants_killed >= 3,
        "test_hollow.py": lambda r: r.passed and r.mutants_killed == 0,
        "test_broken.py": lambda r: r.source_ok and not r.passed,
    }
    ok = True
    for name, expect in expectations.items():
        r = run_tests("python", code, (EX / name).read_text(), muts, cfg)
        good = expect(r)
        ok &= good
        print(f"{'OK ' if good else 'KO '} {name:16} passed={r.passed} "
              f"mutants={r.mutants_killed}/{r.mutants_total} score={r.mutation_score}")
        if not good:
            print(r.log)
    print("Sandbox OK" if ok else "Sandbox KO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
