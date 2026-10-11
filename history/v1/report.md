> Évaluation du 2026-10-11, vLLM (bf16) sur RTX 4090, température 0, **max_tokens 3072** :
> cette limite coupait 7 fichiers sur 25 de TestMaker (comptés « pas de code »).

Évaluation sur 25 modules du split test (communs à : base-gpu, testmaker-gpu, teacher).

| Système | Passe tel quel | Passe après élagage | Tests utiles | Score de mutation | Tests par fichier | Génération |
|---|---|---|---|---|---|---|
| base-gpu (Qwen/Qwen2.5-Coder-1.5B-Instruct) | 0 % | 4 % | 0 % | 0.33 | 4 | 16 s |
| testmaker-gpu (testmaker) | 0 % | 28 % | 12 % | 0.43 | 18 | 26 s |
| teacher (qwen/qwen3-coder-30b-a3b-instruct) | 8 % | 76 % | 56 % | 0.70 | 22 | 85 s |

- **Passe tel quel** : le fichier généré passe sur le code original sans retouche.
- **Passe après élagage** : au plus 30 % de fonctions de test fausses, retirées automatiquement.
- **Tests utiles** : passe (après élagage) et tue au moins 34% des mutants.
- **Score de mutation** : part des bugs introduits détectés, sur les fichiers qui passent.
- **Tests par fichier** : fonctions de test restantes, sur les fichiers qui passent.
