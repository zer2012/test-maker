# test-maker

Distilling of a Qwen code model to have a smaller model capable of writing test code.
Distillation d'un petit LLM (Qwen2.5-Coder) qui écrit des tests unitaires.

Le teacher génère des tests, chaque test est exécuté dans une sandbox Docker (avec mutation testing)
et seuls les tests validés servent à entraîner le student.

## Langages
| Langage | Framework de test | État |
|---|---|---|
| Python | pytest | en cours : génération et validation des données |
| JavaScript | `node:test` | prévu |
| TypeScript | `node:test` | prévu |
| Go | `testing` | prévu |
| Java | JUnit 5 | prévu |
| Kotlin | JUnit 5 | prévu |
| Rust | `#[test]` | prévu |

## Installation (une fois)
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/build_images.sh python          # image Docker de sandbox
ollama pull qwen2.5-coder:7b              # teacher local (~4,7 Go)
.venv/bin/python scripts/selftest_sandbox.py   # doit afficher "Sandbox OK"
```

## Pipeline
```sh
.venv/bin/python scripts/gen_sources.py --lang python --n 20   # modules synthétiques -> data/sources.jsonl
.venv/bin/python scripts/gen_tests.py --lang python             # candidats de tests   -> data/teacher_gen.jsonl
.venv/bin/python scripts/validate.py                            # Docker + mutants     -> data/teacher_val.jsonl
.venv/bin/python scripts/build_sft.py                           # jeu SFT              -> data/sft/*.jsonl
```
Chaque script reprend là où il s'est arrêté. `--provider openrouter` (avec `TEACHER_API_KEY`)
bascule sur Qwen3-Coder-30B-A3B pour le volume ; le teacher local sert à valider la chaîne.

## Entraînement
LoRA sur GPU loué à partir de `data/sft/` (dépendances : `train/requirements.txt`) :
```sh
python train/train_lora.py      # -> checkpoints/testmaker-1.5b/adapter
python train/merge.py           # -> checkpoints/testmaker-1.5b/merged + Modelfile
```
Le modèle fusionné s'installe ensuite dans Ollama sous le nom `testmaker`.

## Évaluation
Le modèle de base, TestMaker et le teacher écrivent des tests pour les modules du split test, jugés
comme à la validation (sandbox, mutants, élagage). Systèmes définis dans `config.yaml` (`eval:`).
```sh
.venv/bin/python scripts/try_model.py 3                  # un module, un modèle, résultat détaillé
.venv/bin/python scripts/evaluate.py                     # comparaison complète -> data/eval/v1/report.md
```

## Historique
Résultats de chaque version entraînée (données, réglages, évaluation) : [history/](history/).
