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
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

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

# Fichiers du tokenizer d'origine. Le LoRA ne modifie pas le vocabulaire : on les recopie tels quels
# plutôt que de les réécrire avec transformers 5, dont le format n'est pas lu par les outils de
# conversion GGUF (llama.cpp) ni par toutes les versions de vLLM.
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def merge(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Applique l'adaptateur, puis ajoute ses corrections dans les poids de base (W += B @ A * alpha / r)."""
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir))


def copy_original_tokenizer(base_model: str, merged_dir: Path) -> None:
    for name in TOKENIZER_FILES:
        hf_hub_download(base_model, name, local_dir=str(merged_dir))


def main() -> None:
    cfg = load_config()
    base_model = cfg["student"]["base_model"]
    out_dir = ROOT / cfg["student"]["output_dir"]
    merged_dir = out_dir / "merged"

    merge(base_model, out_dir / "adapter", merged_dir)
    copy_original_tokenizer(base_model, merged_dir)
    (out_dir / "Modelfile").write_text(MODELFILE)
    print(f"Modèle fusionné : {merged_dir}\nModelfile : {out_dir / 'Modelfile'}")


if __name__ == "__main__":
    main()
