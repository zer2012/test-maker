# Historique des versions de TestMaker

Chaque version a son dossier avec `report.md`, le tableau d'évaluation produit par `scripts/evaluate.py`. Les scores se comparent entre versions
seulement si l'évaluation a été faite dans les mêmes conditions (mêmes modules test, mêmes réglages).

| Version | Date | Exemples train | Passe après élagage (base / TestMaker / teacher) | Tests utiles (base / TestMaker / teacher) | Détails |
|---|---|---|---|---|---|
| v1 | 2026-10-11 | 307 (Python) | 4 % / 28 % / 76 % | 0 % / 12 % / 56 % | [v1](v1/) — max_tokens 3072, TestMaker sous-estimé |

**Tests utiles** = le fichier passe (après élagage) et détecte au moins un tiers des bugs introduits.
