import os
import json
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt


def Start_LoRA_Train(trainer, LoRA_model, tokenizer, loss_store_dir, LoRA_save_dir, output_dir, Load_LoRA_dir=None):
    """
    一键启动训练、自动断点续跑、训练完存 LoRA、出图。

    trainer        : 已配置好的 SFTTrainer 实例
    LoRA_model     : 已注入 LoRA 适配器的模型
    tokenizer      : tokenizer
    loss_store_dir : loss 日志目录（同目录反复调用则自动续接 step 编号）
    LoRA_save_dir  : LoRA adapter 最终保存路径
    output_dir     : HuggingFace Trainer checkpoint 输出目录
    Load_LoRA_dir  : （可选）训练前加载已有 LoRA 权重作为起点，None 则从随机初始化开始
    """
    if Load_LoRA_dir is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(Load_LoRA_dir, "adapter_model.safetensors"))
        set_peft_model_state_dict(LoRA_model, state_dict)
        print(f"✅ LoRA weights loaded from {Load_LoRA_dir}")

    os.makedirs(loss_store_dir, exist_ok=True)

    # 断点续跑检测
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    checkpoint = ckpts[-1] if ckpts else None

    trainer.train(resume_from_checkpoint=checkpoint)

    # 清理 checkpoint 目录
    for c in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(c)

    # 保存 LoRA adapter
    LoRA_model.save_pretrained(LoRA_save_dir)
    tokenizer.save_pretrained(LoRA_save_dir)
    print(f"✅ LoRA saved → {LoRA_save_dir}")

    # 训练完出图
    Loss_Visualisation(loss_store_dir)


def Map_Loss_to_Samples(step, trainer, dataset):
    """
    输入一个 optimizer step 编号，返回该 step 用到的所有训练样本原文。
    注意：dataset 需要有 'text' 字段（未 tokenize 的原始数据集）。
    """
    effective_batch = trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps
    start = (step - 1) * effective_batch
    end   = step * effective_batch

    dataloader = trainer.get_train_dataloader()
    all_indices = list(dataloader.sampler)

    if end > len(all_indices):
        print(f"⚠️  step {step} 超出范围，数据集共 {len(all_indices)} 个样本，最大 step = {len(all_indices) // effective_batch}")
        return

    step_indices = all_indices[start:end]
    sep = "\n\n" + "=" * 25 + "\n\n"
    samples = [dataset[int(i)]["text"] for i in step_indices]

    print(f"📌 Step {step}  |  effective_batch={effective_batch}  |  样本索引 {start}~{end-1}\n")
    print(sep.join(samples))


def Loss_Visualisation(loss_dir, save_path=None):
    """从 loss_dir/loss_history.jsonl 读取训练记录，保存 loss 曲线图到文件。"""
    loss_file = os.path.join(loss_dir, "loss_history.jsonl")
    if not os.path.exists(loss_file):
        print(f"⚠️  找不到 loss 文件：{loss_file}")
        return

    records = []
    with open(loss_file) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if not r.get("__sep__"):
                    records.append(r)

    if not records:
        print("⚠️  loss 文件为空，检查训练是否正常完成。")
        return

    df = pd.DataFrame(records)
    if "Loss" not in df.columns:
        print("⚠️  找不到 Loss 列，检查 loss 文件格式。")
        return

    p1 = df[["Step", "Loss"]].dropna().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(p1["Step"], p1["Loss"], alpha=0.5, color="#3498db", label="Raw Loss")
    if len(p1) > 20:
        p1["smooth"] = p1["Loss"].rolling(window=10).mean()
        ax.plot(p1["Step"], p1["smooth"], color="#2c3e50", linewidth=2, label="Smoothed (MA-10)")

    ax.set_title("Training Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(loss_dir, "loss_final.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()

    print(f"📊 Loss curve saved → {save_path}")
    print(f"📊 Start loss: {p1['Loss'].iloc[0]:.4f} → End loss: {p1['Loss'].iloc[-1]:.4f}  |  Total steps: {len(p1)}")
