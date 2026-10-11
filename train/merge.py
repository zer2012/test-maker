"""Fusionne l'adaptateur LoRA dans le modèle de base -> <output_dir>/merged + Modelfile pour Ollama.

À lancer sur le pod après l'entraînement, depuis la racine du repo :
    python train/merge.py
Le dossier `merged` est un modèle complet et autonome (même taille que Qwen2.5-Coder-1.5B).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from testmaker.common import load_config

# Modèle de chat ChatML de Qwen, écrit explicitement pour qu'Ollama formate les messages exactement
# comme à l'entraînement. Température 0 : on veut la réponse la plus probable, reproductible.
MODELFILE = '''FROM ./merged
TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER num_ctx 8192
'''


def main() -> None:
    cfg = load_config()
    out_dir = ROOT / cfg["student"]["output_dir"]
    base = AutoModelForCausalLM.from_pretrained(cfg["student"]["base_model"], dtype=torch.bfloat16)
    # Applique l'adaptateur, puis ajoute ses corrections dans les poids de base (W += B @ A * alpha / r).
    model = PeftModel.from_pretrained(base, str(out_dir / "adapter")).merge_and_unload()
    model.save_pretrained(str(out_dir / "merged"))
    AutoTokenizer.from_pretrained(str(out_dir / "adapter")).save_pretrained(str(out_dir / "merged"))
    (out_dir / "Modelfile").write_text(MODELFILE)
    print(f"Modèle fusionné : {out_dir / 'merged'}\nModelfile : {out_dir / 'Modelfile'}")


if __name__ == "__main__":
    main()
