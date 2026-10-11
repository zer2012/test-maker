"""Utilitaires partagés par tous les scripts : config, fichiers JSONL, hachage, split, extraction de code."""

import hashlib
import json
import os
import re
from pathlib import Path

import yaml

# Racine du projet (le dossier qui contient `testmaker/` et `config.yaml`).
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    """Charge `config.yaml` et rend tous les chemins de `paths:` absolus."""
    with open(path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Chemins relatifs à la racine du projet, quel que soit le dossier courant.
    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def read_jsonl(path) -> list[dict]:
    """Lit un fichier JSONL (un objet JSON par ligne) ; fichier absent = liste vide."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record: dict) -> None:
    """Ajoute un enregistrement en fin de fichier.

    Les scripts écrivent au fil de l'eau : en cas d'interruption, rien n'est perdu et
    la relance saute les ids déjà présents.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        # ensure_ascii=False garde les accents et symboles lisibles dans le fichier.
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path, records) -> None:
    """Réécrit entièrement un fichier JSONL (utilisé pour les sorties finales, ex. data/sft)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def code_hash(code: str) -> str:
    """Empreinte du code insensible aux espaces, pour repérer les doublons."""
    # Toute suite d'espaces/tabulations/retours à la ligne devient un seul espace.
    normalized = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_of(record_id: str) -> str:
    """Split déterministe par id : 90% train, 5% val, 5% test."""
    # Le hash d'un id donne toujours le même nombre : un exemple reste dans le même split
    # d'une exécution à l'autre, même si le dataset grossit (pas de fuite train -> test).
    bucket = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 90 else "val" if bucket < 95 else "test"


def extract_code(text: str) -> str:
    """Extrait le plus long bloc de code Markdown d'une réponse de LLM."""
    # ``` + langage optionnel (python, c++, c#, objective-c...) + fin de ligne, puis le contenu
    # jusqu'au ``` suivant. re.S permet à `.` de traverser les retours à la ligne ;
    # `.*?` (non gourmand) s'arrête à la première clôture.
    blocks = re.findall(r"```[\w+#-]*[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len).strip()
    # Pas de bloc complet : on accepte le texte brut, sauf s'il contient un bloc mal fermé
    # (réponse probablement tronquée par max_tokens).
    return text.strip() if "```" not in text else ""
