# Multi_Turn_LoRA

基于 Unsloth + SFTTrainer 的多轮对话 LoRA 微调框架，针对 Qwen2.5-7B-Instruct。

---

## 快速开始

```bash
cd /home/levizenith/SednaAI/Multi_Turn_LoRA && python LoRA_Execution.py
```

---

## 目录结构

```
Multi_Turn_LoRA/
├── LoRA_Execution.py       配置文件 + 训练入口（只需改这里）
├── pipeline.py             完整训练流程
├── callbacks.py            训练回调（进度条、日志、loss 图）
├── mask_loss.py            Mask Loss 函数库
├── train.py                训练启动 / loss 可视化 / 样本追溯
├── Multi_Turn_LoRA.ipynb   原始交互式 notebook（保留备用）
└── LoRA_Visualization/
    └── loss_curve.png      训练中每隔 N 步自动覆盖更新
```

训练过程中自动生成：

```
training_progress/
└── <数据集名>/
    ├── loss_history.jsonl  完整训练日志
    └── loss_final.png      训练结束后的完整 loss 曲线
```

---

## 各文件说明

### `LoRA_Execution.py`

唯一需要改动的文件。所有训练参数集中在这里，改完直接执行。

| 参数 | 说明 |
|---|---|
| `model_path` | 基础模型路径 |
| `dataset_path` | 训练数据路径（MultiTurnSplit JSON）|
| `checkpoint_dir` | HuggingFace Trainer 断点续跑的临时 checkpoint 目录，训练完自动清空 |
| `loss_store_dir` | loss 日志目录，同路径续跑时 step 编号自动续接 |
| `lora_save_dir` | 训练完成后 LoRA adapter 的保存路径 |
| `load_lora_dir` | 从已有 LoRA 权重继续训练，`None` 则从随机初始化 |
| `vis_dir` | loss 曲线图的覆盖保存目录 |

---

### `pipeline.py`

`run(cfg)` 函数，顺序执行：加载模型 → 注入 LoRA → 加载数据 → 构建 Trainer → Mask Loss → 启动训练。由 `LoRA_Execution.py` 调用，通常不需要改动。

---

### `callbacks.py`

四个训练回调：

**`PrettyTableCallback`** — 每 step 打印一行训练记录（step / loss / GPU 显存），并持久化写入 `loss_history.jsonl`。支持断点续跑时的日志续接。

**`TQDMProgressCallback`** — 终端 tqdm 进度条，实时显示训练进度和当前 loss。

**`CheckpointCounterCallback`** — 每次保存 checkpoint 时打印计数提示。

**`LiveLossPlotCallback`** — 后台线程每隔 `plot_interval` 步生成一张 loss 曲线图，覆盖保存到 `LoRA_Visualization/loss_curve.png`，随时可以打开查看。

---

### `mask_loss.py`

**`Mask_Loss(trainer, tokenizer, raw_train_dataset)`** — 对训练集做 token 级 mask，只对每条样本的最后一轮 assistant 回复计算 loss，其余位置全部打 -100。使用自定义 collator 替换 Trainer 默认的数据处理。

---

### `train.py`

**`Start_LoRA_Train(...)`** — 断点续跑检测 → 启动训练 → 清理 checkpoint → 保存 LoRA → 输出 loss 曲线图。

**`Loss_Visualisation(loss_dir)`** — 读取 `loss_history.jsonl`，生成带移动平均的 loss 曲线，保存为 `loss_final.png`。

**`Map_Loss_to_Samples(step, trainer, dataset)`** — 调试工具，输入 step 编号，返回该 step 用到的所有原始训练样本文本。

---

## 断点续跑

训练中断后直接重新执行同一命令即可，框架自动检测 `checkpoint_dir` 下的断点并续接，loss 日志和曲线图也会连续显示，无需手动处理。
