from pipeline import run

# ================================================================
# CONFIG
# ================================================================

model_path     = "/home/levizenith/SednaAI/Qwen2.5-7B-Instruct-bnb-4bit"
dataset_path   = "/home/levizenith/SednaAI/地理探索_MultiTurnSplit.json"
checkpoint_dir = "/mnt/f/Programming/DS_ML_DL/Transformer/Multi_Turn_LoRA"
loss_store_dir = "training_progress/地理探索"
lora_save_dir  = "/home/levizenith/SednaAI/LoRA_地理探索"
load_lora_dir  = None   # 填路径则从已有 LoRA 继续，None 则随机初始化

vis_dir        = "LoRA_Visualization"
max_seq_length = 8000

lora_r         = 16
lora_alpha     = 16
lora_dropout   = 0

num_epochs     = 1
batch_size     = 1
grad_accum     = 8
learning_rate  = 2e-4
warmup_steps   = 5
weight_decay   = 0.01
lr_scheduler   = "linear"
neftune_alpha  = 1
save_steps     = 50
seed           = 524
plot_interval  = 10   # 每隔多少步刷新一次 loss 图

# ================================================================

run(locals())
