# Multi_Turn_LoRA — 多轮对话 LoRA 训练文档

`Multi_Turn_LoRA.ipynb` 是 SednaAI 的多轮对话微调训练 notebook，基于 Unsloth + SFTTrainer 对 Qwen2.5-7B-Instruct-bnb-4bit 做 LoRA 训练。前 8 个板块覆盖从模型加载到训练完成的完整流程。

## 训练流水线概览

```
Load Model  →  Load JSON  →  Inject Prompt  →  Strategy A 展开
    →  Prompt 可视化  →  Chat Template  →  SFTTrainer 初始化
    →  Mask Loss  →  GPU 监控  →  Phase 1 训练  →  Loss 可视化
```

---

## 板块详解

### 1. Dependencies

导入全部依赖：

| 类别 | 库 |
|---|---|
| 模型与训练 | `unsloth`（FastLanguageModel）、`transformers`、`trl`（SFTTrainer）、`peft` |
| 数据 | `datasets`（Dataset）、`pandas` |
| 计算 | `torch` |
| 可视化 | `plotly`、`matplotlib` |
| 工具 | `json`、`random`、`hashlib`、`time` |

---

### 2. Load Model

**`Load_Model(model_path, max_seq_length, dtype=None, load_in_4bit=True)`**

一次调用完成模型加载与 LoRA 注入两步。

```python
model, tokenizer, max_seq_length = Load_Model(
    model_path = "/home/levizenith/SednaAI/Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = 8000,
)
```

内部流程：

**Step 1** — `FastLanguageModel.from_pretrained`：4bit 量化加载，dtype=float16，device_map={"": 0}，立即调用 `for_training()` 开启训练模式。

**Step 2** — `FastLanguageModel.get_peft_model`：注入 LoRA adapter，参数如下：

| 参数 | 值 | 说明 |
|---|---|---|
| `r` | 8 | LoRA 低秩矩阵维度 |
| `lora_alpha` | 8 | 缩放系数（alpha/r = 1，不放大梯度） |
| `target_modules` | q/k/v/o/gate/up/down × 7 | QKVO + SwiGLU 全部线性层 |
| `lora_dropout` | 0 | 不加 dropout |
| `bias` | none | 不训练偏置项 |
| `use_gradient_checkpointing` | "unsloth" | Unsloth 优化的梯度检查点，节省 VRAM |
| `random_state` | 524 | 固定随机种子 |

---

### 3. Load The Dataset & Apply Chat Template

#### 数据加载

读取 Skeleton 生成阶段产出的 JSON 文件：

```python
file_path = "/mnt/f/Programming/DS_ML_DL/Transformer/Multi_Turn_LoRA/地图探索_Skeleton.json"
with open(file_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
```

格式：`{话题名: [{"role": "system/user/assistant", "content": "..."}]}`

#### Inject Prompt

**`Inject_Prompt(xlsx_path, raw_data)`** — 将 RAG 知识内容注入 system prompt。

```python
xlsx_path = "/mnt/f/Programming/DS_ML_DL/Transformer/Multi_Turn_LoRA/地图探索_Question.xlsx"
Inject_Prompt(xlsx_path, raw_data)
```

- 读取 xlsx 第 A 列（话题名）与第 C 列（RAG 知识内容）
- 以话题名为 key 做精确匹配，命中则将知识内容填入 `_BASE_PROMPT` 的 `{rag_content}` 占位符
- 原地修改 `raw_data` 中每条对话的 system message content
- 打印命中数 / 未命中列表

#### Strategy A 展开

将每条 N 轮对话展开为 N 个训练样本，第 i 个样本只训练第 i 轮 assistant 的回复：

```
原始对话（N 轮）  →  样本 1: [sys][user1][ass1]
                     样本 2: [sys][user1][ass1][user2][ass2]
                     ...
                     样本 N: [sys][user1][ass1]...[userN][assN]
```

- 先前各轮的 assistant 回复在训练时全部 mask（labels=-100），不产生梯度
- 每个样本只对最后一轮 assistant 计算 loss
- 100 条话题 × 平均 9.45 轮 = **945 条训练样本**

#### Prompt Count Visualisation

**`prompt_count_visualization(expanded_data, tokenizer)`** — 统计每条样本的 prompt token 数（即不含最后一轮 assistant 的部分）并生成交互式分布图。

- 用 Qwen tokenizer 计算 token 数
- Plotly 直方图，标出 min（绿色虚线）、max（红色虚线）、mean（橙色点线）
- 用于评估序列长度分布，确认 `max_seq_length` 设置合理

#### Chat Template

```python
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
all_texts = [
    tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    for conv in expanded_data
]
random.shuffle(all_texts)
dataset = Dataset.from_dict({"text": all_texts})
```

shuffle 的目的是避免同一条原始对话的多个展开样本连续出现在同一 gradient accumulation batch 里。

---

### 4. Training theme

本板块说明两阶段训练设计与 COT 机制。

#### COT（思维链）

Sedna 的 assistant 回复分为两部分：
- **COT 段**：`<think>` 标签内的分析推理过程
- **Answer 段**：实际输出给用户的最终回复

两者在 tokenizer 层面通过特征字符串区分，`_assistant_content_spans` 与 `_answer_only_spans` 分别定位完整 assistant 内容区域和纯 answer 区域。

#### 两阶段训练

| 阶段 | `cot_ratio` | 训练内容 | 意图 |
|---|---|---|---|
| Phase 1 | 1.0 | 所有样本训练 COT + Answer | 先学会如何推理和组织回答 |
| Phase 2 | 0.2 | 20% 样本训练 COT + Answer，80% 仅训练 Answer | 巩固输出风格，同时保留部分推理能力 |

`_select_cot_indices(n_samples, cot_ratio, seed)` 使用 hash-based jittered stratified sampling 选取哪些样本参与 COT 训练，保证分布均匀但排列不规则。

---

### 5. SFTTrainer & Mask Loss

#### SFTTrainer 配置

```python
split = dataset.train_test_split(test_size=0.1, seed=524)
trainer = SFTTrainer(
    model = LoRA_model,
    tokenizer = tokenizer,
    train_dataset = split["train"],
    eval_dataset  = split["test"],
    dataset_text_field = "text",
    max_seq_length = 8000,
    packing = False,
    args = TrainingArguments(
        num_train_epochs           = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,   # 有效 batch size = 8
        learning_rate              = 2e-4,
        optim                      = "adamw_8bit",
        weight_decay               = 0.01,
        warmup_steps               = 5,
        lr_scheduler_type          = "linear",
        neftune_noise_alpha        = 1,    # 随机扰动嵌入层，提升泛化
        fp16                       = True,
    )
)
raw_train_dataset = trainer.train_dataset  # 保存原始未 mask 数据集，供 Mask_Loss 复用
```

#### Qwen: Manual Mask Loss

SFTTrainer 默认对所有 token 计算 loss（包括 system / user / 历史 assistant）。本板块手动实现字符级 token masking，只训练指定 span。

**核心函数：**

`_build_labels_from_spans(text, tokenizer, spans)` — 对整段文本 tokenize，将 spans 范围外的 token labels 全部设为 -100。使用 `offset_mapping` 做字符到 token 的精确对齐。

`_assistant_content_spans(text)` — 用 Qwen 的 `<|im_start|>assistant` / `<|im_end|>` 边界定位所有 assistant 内容段。

`_answer_only_spans(text)` — 在 assistant 段内进一步跳过 `<think>...</think>` 部分，只返回最终答案段。

`_last_assistant_span(text)` / `_last_answer_only_span(text)` — 从上述函数结果中只取**最后一个** span，适配 Strategy A（每个样本只训练最后一轮）。

**`_make_collator(tokenizer)`** — 自定义 padding collator，对 pre-tokenized 的 `input_ids / attention_mask / labels` 做动态 padding（右填充到 batch 内最长序列）。

**`Mask_Loss(trainer, tokenizer, cot_ratio=1.0, seed=524, num_proc=4)`** — 对外接口：

```python
Mask_Loss(trainer, tokenizer, cot_ratio=1.0)
```

内部流程：
1. 将 `trainer.train_dataset` 重置为 `raw_train_dataset`（原始文本格式）
2. 用 `_select_cot_indices` 随机选取 COT 样本索引
3. `dataset.map(map_fn)` 逐条 tokenize + masking，产出 `input_ids / attention_mask / labels`
4. 替换 `trainer.train_dataset` 为 masked 版本
5. 替换 `trainer.data_collator` 为 `_make_collator`
6. 设 `remove_unused_columns=False` 防止 SFTTrainer 过滤预处理字段

**`check_mask_loss(index, trainer, tokenizer)`** — 抽查某条样本，打印哪些 token 是可训练的（label != -100），验证 masking 逻辑是否正确。

---

### 6. GPU Supervise

训练前和训练中实时监控 GPU 显存：

| 指标 | 含义 |
|---|---|
| `gpu_alloc(MB)` | 已分配显存（模型 + 数据 + 激活值） |
| `gpu_reserved(MB)` | PyTorch 预留显存（含碎片） |
| `gpu_cap(MB)` | GPU 总显存容量 |

通过 `TrainerCallback` 在每个 logging step 打印以上三个指标，配合训练 loss 一起输出。

---

### 7. Start Training

```python
# Phase 1：COT + Answer 全量训练
Mask_Loss(trainer, tokenizer, cot_ratio=1.0)
trainer.train()
phase1_logs = trainer.state.log_history.copy()

# Phase 2（可选）：Answer-only 为主
Mask_Loss(trainer, tokenizer, cot_ratio=0.2)
trainer.train()
```

Unsloth 会自动检测 VRAM 压力并智能 offload 梯度（"Unsloth: Will smartly offload gradients to save VRAM!"），在 8GB 显卡上可正常运行。

每 Phase 独立调用 `Mask_Loss` 的原因是它会 reset `train_dataset`，重新 tokenize + masking，确保两个阶段使用完全不同的 label 配置。

---

### 8. Loss Visualisation

```python
# Phase 1 Loss Visualisation
```

用 matplotlib 绘制训练 loss 曲线：

- 蓝色半透明折线：每 step 的原始 loss
- 深色实线：MA-10 平滑曲线（步数 > 20 时启用）
- 横轴：training steps，纵轴：loss
- 标题注明当前 learning rate 与训练阶段
- 结尾打印 `📊 Phase 1 | Start loss: X.XXXX → End loss: X.XXXX`

---

## 关键路径

| 文件 | 说明 |
|---|---|
| `Multi_Turn_LoRA.ipynb` | 本训练 notebook |
| `Qwen2.5-7B-Instruct-bnb-4bit/` | 基础模型（4bit 量化） |
| `unsloth_compiled_cache/` | Unsloth CUDA kernel 编译缓存，max_seq_length 变化时会重新编译 |
| `地图探索_Skeleton.json` | Skeleton 阶段产出，包含每轮 user 消息末尾的 `【回答节点推进数：N】` 注释 |
| `地图探索_Question.xlsx` | RAG 知识库，A 列话题名、C 列知识内容 |

## 注意事项

**max_seq_length**：在 `Load_Model` 调用和 `SFTTrainer` 中保持一致（当前推荐 `8000`）。修改该值会触发 Unsloth CUDA kernel 重新编译（缓存在 `unsloth_compiled_cache/`），编译完成前不要中断。若编译缓存损坏导致 OOM，删除 `~/SednaAI/unsloth_compiled_cache/` 后重启 kernel 重新编译。

**Kernel Restart**：修改 `max_seq_length` 或任何模型加载参数后，必须 Restart Kernel & Run All，单独重跑某个 cell 不会让已加载的模型感知到变更。

**【回答节点推进数：N】**：Skeleton JSON 中每轮 user 消息末尾附带的节点推进数标注，由 `skeleton_generation.py` 在骨架生成完成后 Python 注入，保证训练信号与生成约束一致。推理时由后端应用层按规则注入（第一轮固定 3，后续按策略决定）。
