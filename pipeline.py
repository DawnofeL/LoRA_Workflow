import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(cfg):
    import json
    import random
    import time
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from transformers.trainer_callback import ProgressCallback, PrinterCallback
    from datasets import Dataset

    from callbacks import PrettyTableCallback, TQDMProgressCallback, CheckpointCounterCallback, LiveLossPlotCallback
    from mask_loss  import Mask_Loss
    from train      import Start_LoRA_Train

    # ---- Load Model ----
    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg["model_path"],
        max_seq_length = cfg["max_seq_length"],
        dtype          = torch.float16,
        load_in_4bit   = True,
        device_map     = {"": 0},
    )
    FastLanguageModel.for_training(model)
    print(f"✅ Model loaded from {cfg['model_path']}  ({time.perf_counter() - t0:.1f}s)")

    # ---- Inject LoRA ----
    LoRA_model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg["lora_r"],
        target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha                 = cfg["lora_alpha"],
        lora_dropout               = cfg["lora_dropout"],
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = cfg["seed"],
        use_rslora                 = False,
        loftq_config               = None,
    )

    # ---- Load Data ----
    with open(cfg["dataset_path"], "r", encoding="utf-8") as f:
        expanded_data = json.load(f)
    print(f"✅ Dataset loaded: {len(expanded_data)} samples")

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    all_texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in expanded_data
    ]
    random.shuffle(all_texts)
    dataset = Dataset.from_dict({"text": all_texts})

    # ---- Trainer Setup ----
    os.makedirs(cfg["loss_store_dir"], exist_ok=True)

    trainer = SFTTrainer(
        model              = LoRA_model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = cfg["max_seq_length"],
        packing            = False,
        dataset_num_proc   = 4,
        callbacks=[
            TQDMProgressCallback(refresh_interval=1.0),
            LiveLossPlotCallback(loss_dir=cfg["loss_store_dir"], vis_dir=cfg["vis_dir"], plot_interval=cfg["plot_interval"]),
            PrettyTableCallback(loss_dir=cfg["loss_store_dir"]),
            CheckpointCounterCallback(),
        ],
        args = TrainingArguments(
            num_train_epochs            = cfg["num_epochs"],
            per_device_train_batch_size = cfg["batch_size"],
            gradient_accumulation_steps = cfg["grad_accum"],
            warmup_steps                = cfg["warmup_steps"],
            optim                       = "adamw_8bit",
            weight_decay                = cfg["weight_decay"],
            lr_scheduler_type           = cfg["lr_scheduler"],
            learning_rate               = cfg["learning_rate"],
            neftune_noise_alpha         = cfg["neftune_alpha"],
            bf16                        = False,
            logging_steps               = 1,
            report_to                   = "none",
            disable_tqdm                = True,
            seed                        = cfg["seed"],
            output_dir                  = cfg["checkpoint_dir"],
            save_strategy               = "steps",
            save_steps                  = cfg["save_steps"],
            save_total_limit            = 1,
            gradient_checkpointing      = True,
        ),
    )

    # 移除 HF 默认进度回调，避免与自定义 callback 冲突
    trainer.callback_handler.callbacks = [
        c for c in trainer.callback_handler.callbacks
        if type(c) not in (ProgressCallback, PrinterCallback)
    ]

    raw_train_dataset = trainer.train_dataset

    # ---- Mask Loss ----
    Mask_Loss(trainer, tokenizer, raw_train_dataset)

    # ---- Train ----
    Start_LoRA_Train(
        trainer        = trainer,
        LoRA_model     = LoRA_model,
        tokenizer      = tokenizer,
        loss_store_dir = cfg["loss_store_dir"],
        LoRA_save_dir  = cfg["lora_save_dir"],
        output_dir     = cfg["checkpoint_dir"],
        Load_LoRA_dir  = cfg.get("load_lora_dir"),
    )
