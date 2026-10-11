"""Entraîne le student (Qwen2.5-Coder-1.5B) avec LoRA sur data/sft -> <output_dir>/adapter.

À lancer sur le pod GPU, depuis la racine du repo :
    python train/train_lora.py --max-steps 2     # essai rapide : vérifie que tout se charge
    python train/train_lora.py                   # entraînement complet
    python train/train_lora.py --qlora           # variante 4 bits pour petit GPU

Les réglages viennent de config.yaml (`student:` et `sft:`).
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from testmaker.common import load_config


def load_split(path: Path, tokenizer, max_length: int):
    """Charge un fichier SFT et écarte les exemples trop longs (au lieu de les tronquer)."""
    ds = load_dataset("json", data_files=str(path), split="train")
    # TRL n'a besoin que de ces deux colonnes ; `id` et `lang` resteraient sinon dans les batchs.
    ds = ds.select_columns(["prompt", "completion"])

    def n_tokens(ex) -> int:
        # Même mise en forme que celle que TRL appliquera : modèle de chat de Qwen sur prompt + réponse.
        text = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    before = len(ds)
    ds = ds.filter(lambda ex: n_tokens(ex) <= max_length)
    print(f"{path.name}: {len(ds)} exemples ({before - len(ds)} écartés, > {max_length} tokens)")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlora", action="store_true", help="force le chargement 4 bits (sinon : config sft.qlora)")
    ap.add_argument("--max-steps", type=int, default=-1, help="limite le nombre de pas (essai rapide)")
    args = ap.parse_args()

    cfg = load_config()
    sft, student = cfg["sft"], cfg["student"]
    data_dir = Path(cfg["paths"]["sft_dir"])
    out_dir = ROOT / student["output_dir"]
    qlora = args.qlora or sft.get("qlora", False)

    tokenizer = AutoTokenizer.from_pretrained(student["base_model"])
    train_ds = load_split(data_dir / "train.jsonl", tokenizer, sft["max_length"])
    eval_ds = load_split(data_dir / "val.jsonl", tokenizer, sft["max_length"])

    # LoRA sur toutes les couches linéaires (attention + MLP) : le réglage standard pour du SFT.
    peft_config = LoraConfig(r=sft["lora_r"], lora_alpha=sft["lora_alpha"], lora_dropout=sft["lora_dropout"],
                             target_modules="all-linear", task_type="CAUSAL_LM")
    # QLoRA : le modèle de base est figé en 4 bits (NF4), seuls les adaptateurs LoRA restent en bf16.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if qlora else None

    # Évaluation deux fois par epoch : assez de points pour voir un surapprentissage (val qui remonte).
    steps_per_epoch = math.ceil(len(train_ds) / (sft["batch_size"] * sft["grad_accum"]))
    eval_steps = max(1, steps_per_epoch // 2)

    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=sft["epochs"],
        max_steps=args.max_steps,
        learning_rate=sft["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=sft["warmup"],                    # < 1 : interprété comme une part des pas
        per_device_train_batch_size=sft["batch_size"],
        per_device_eval_batch_size=sft["batch_size"],
        gradient_accumulation_steps=sft["grad_accum"],
        gradient_checkpointing=True,                   # recalcule les activations : moins de mémoire, ~30% plus lent
        bf16=True,
        max_length=sft["max_length"],
        # Avec un dataset prompt/completion, TRL ne calcule la perte que sur la completion (le test).
        completion_only_loss=True,
        # TRL charge sinon le modèle en float32 : deux fois plus lourd sans gain.
        model_init_kwargs={"dtype": "bfloat16"},
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,                   # garde le checkpoint de plus faible perte de validation
        metric_for_best_model="eval_loss",
        logging_steps=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=student["base_model"],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))               # avec PEFT : n'enregistre que l'adaptateur (~74 M paramètres, 150-300 Mo)
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nPerte de validation finale : {trainer.evaluate()['eval_loss']:.4f}")
    print(f"Adaptateur LoRA : {adapter_dir}")


if __name__ == "__main__":
    main()
