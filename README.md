# test-maker

Distilling of a Qwen code model to have a smaller model capable of writing test code.
Distillation d'un petit LLM (Qwen2.5-Coder) qui écrit des tests unitaires.
État actuel : **étapes 1 à 4 en Python, en local** (teacher via Ollama ou OpenRouter, validation Docker).

## Installation (une fois)
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/build_images.sh python          # image Docker de sandbox
ollama pull qwen2.5-coder:7b              # teacher local (~4,7 Go)
.venv/bin/python scripts/selftest_sandbox.py   # doit afficher "Sandbox OK"
```

## Pipeline
```sh
.venv/bin/python scripts/gen_sources.py --lang python --n 20   # 1. modules synthétiques -> data/sources.jsonl
.venv/bin/python scripts/gen_tests.py --lang python             # 2. candidats de tests   -> data/teacher_gen.jsonl
.venv/bin/python scripts/validate.py                            # 3. Docker + mutants     -> data/teacher_val.jsonl
.venv/bin/python scripts/build_sft.py                           # 4. jeu SFT              -> data/sft/*.jsonl
```
Chaque script reprend là où il s'est arrêté. `--provider openrouter` (avec `TEACHER_API_KEY`)
bascule sur Qwen3-Coder-30B-A3B pour le volume ; le teacher local sert à valider la chaîne.
