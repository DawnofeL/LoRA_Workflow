import os
import json
import glob
import threading
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from transformers import TrainerCallback


def _load_history_up_to_ckpt(loss_file, output_dir):
    """
    读取 loss_history.jsonl，只保留 <= checkpoint step 的记录。
    返回 (ckpt_step, filtered_records)。
    如果没有 checkpoint 或 loss 文件，返回 (None, [])。
    """
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    if not ckpts or not os.path.exists(loss_file):
        return None, []
    ckpt_step = int(ckpts[-1].rsplit("-", 1)[-1])

    all_records = []
    with open(loss_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            all_records.append(r)

    # 只保留 checkpoint 之前的数据行，以及之前的分隔行
    filtered = []
    for r in all_records:
        if r.get("__sep__"):
            filtered.append(r)
        elif r["Step"] <= ckpt_step:
            filtered.append(r)

    return ckpt_step, filtered


# ============================================================
# Pretty Table Callback（终端 print + JSONL 持久化 + GPU 显存监控）
# ============================================================

class PrettyTableCallback(TrainerCallback):
    def __init__(self, loss_dir):
        self.records = []
        self._step_offset = 0
        self.loss_dir = loss_dir
        self._loss_file = os.path.join(loss_dir, "loss_history.jsonl")
        os.makedirs(loss_dir, exist_ok=True)

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
        except Exception:
            self._nvml_ok = False

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self._gpu_total_mb = props.total_memory / 1024**2
        else:
            self._gpu_total_mb = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.records = []

        if state.global_step > 0:
            # 断点续跑：加载到 checkpoint step，追加续跑分隔行
            ckpt_step, filtered = _load_history_up_to_ckpt(self._loss_file, args.output_dir)
            if ckpt_step is not None and filtered:
                self.records = filtered
                with open(self._loss_file, "w") as f:
                    for r in filtered:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_resumes = sum(1 for r in self.records if r.get("__sep__"))
                session_num = n_resumes + 1
                sep = {"__sep__": True, "label": f"第{session_num}次断点续跑（从 step {ckpt_step} 继续）"}
                self.records.append(sep)
                with open(self._loss_file, "a") as f:
                    f.write(json.dumps(sep, ensure_ascii=False) + "\n")
                print(f"\n━━━ 第{session_num}次断点续跑（从 step {ckpt_step} 继续）━━━\n")
            self._step_offset = 0
        else:
            # 全新训练：文件存在则续接 step 编号
            if os.path.exists(self._loss_file) and os.path.getsize(self._loss_file) > 0:
                last_step = 0
                with open(self._loss_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        self.records.append(r)
                        if not r.get("__sep__"):
                            last_step = max(last_step, r.get("Step", 0))
                self._step_offset = last_step
            else:
                self._step_offset = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        control.should_log = False
        control.should_evaluate = False

        if self._nvml_ok:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                used_mb  = info.used  / 1024**2
                total_mb = info.total / 1024**2
            except Exception:
                used_mb, total_mb = 0, self._gpu_total_mb
        elif torch.cuda.is_available():
            used_mb  = torch.cuda.memory_reserved() / 1024**2
            total_mb = self._gpu_total_mb
        else:
            used_mb, total_mb = 0, 0

        def fmt(x):
            return float(f"{x:.4g}")

        record = {
            "Step":          state.global_step + self._step_offset,
            "Loss":          fmt(logs["loss"]),
            "gpu_used(MB)":  fmt(used_mb),
            "gpu_total(MB)": fmt(total_mb),
        }

        self.records.append(record)

        with open(self._loss_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  step {record['Step']:>5}  |  loss {record['Loss']:.4f}  |  gpu {record['gpu_used(MB)']:.0f}/{record['gpu_total(MB)']:.0f} MB")


# ============================================================
# Progress Bar Callback（tqdm 终端进度条）
# ============================================================

class TQDMProgressCallback(TrainerCallback):
    def __init__(self, refresh_interval=1.0):
        self._pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._pbar = tqdm(
            total=state.max_steps,
            initial=state.global_step,
            desc="Training",
            dynamic_ncols=True,
        )
        control.should_log = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._pbar is not None and logs and "loss" in logs:
            self._pbar.set_postfix({"loss": f"{logs['loss']:.4f}"})
        control.should_log = False
        control.should_display = False

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ============================================================
# Checkpoint Save Counter Callback
# ============================================================

class CheckpointCounterCallback(TrainerCallback):
    def __init__(self):
        self.n = 0

    def on_save(self, args, state, control, **kwargs):
        self.n += 1
        print(f"\n✅ checkpoint更新次数：{self.n} | step={state.global_step}\n")


# ============================================================
# Live Loss Plot Callback（后台线程渲染，PNG 覆盖保存到 vis_dir）
# ============================================================

class LiveLossPlotCallback(TrainerCallback):
    def __init__(self, loss_dir, vis_dir, plot_interval=10):
        self._loss_file = os.path.join(loss_dir, "loss_history.jsonl")
        self._vis_path  = os.path.join(vis_dir, "loss_curve.png")
        os.makedirs(vis_dir, exist_ok=True)
        self._plot_interval = plot_interval
        self._records = []
        self._rendering = False
        self._step_offset = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._records = []
        self._rendering = False

        if state.global_step > 0:
            ckpt_step, filtered = _load_history_up_to_ckpt(self._loss_file, args.output_dir)
            if ckpt_step is not None:
                for r in filtered:
                    if not r.get("__sep__"):
                        self._records.append({"Step": r["Step"], "Loss": r["Loss"]})
            self._step_offset = 0
        else:
            if os.path.exists(self._loss_file) and os.path.getsize(self._loss_file) > 0:
                last_step = 0
                with open(self._loss_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if not r.get("__sep__"):
                            self._records.append({"Step": r["Step"], "Loss": r["Loss"]})
                            last_step = max(last_step, r["Step"])
                self._step_offset = last_step
            else:
                self._step_offset = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        actual_step = state.global_step + self._step_offset
        self._records.append({
            "Step": actual_step,
            "Loss": float(f"{logs['loss']:.4g}"),
        })

        if actual_step % self._plot_interval != 0:
            return

        if self._rendering:
            return

        records_snapshot = list(self._records)
        vis_path = self._vis_path

        def _bg():
            self._rendering = True
            try:
                self._save_plot(records_snapshot, vis_path)
            except Exception:
                pass
            finally:
                self._rendering = False

        threading.Thread(target=_bg, daemon=True).start()

    def _save_plot(self, records, vis_path):
        steps  = [r["Step"] for r in records]
        losses = [r["Loss"] for r in records]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#1e2127")
        ax.set_facecolor("#1e2127")
        ax.plot(steps, losses, alpha=0.45, color="#3498db", linewidth=1, label="Raw Loss")

        if len(losses) >= 10:
            smooth = pd.Series(losses).rolling(window=10, min_periods=1).mean().tolist()
            ax.plot(steps, smooth, color="#e05c5c", linewidth=1.8, label="Smoothed (MA-10)")

        title = f"Training Loss — step {steps[-1]}" if steps else "Training Loss"
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Steps", color="#aaa", fontsize=9)
        ax.set_ylabel("Loss",  color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(facecolor="#2b2f38", labelcolor="white", fontsize=8, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.25, color="#555")
        fig.tight_layout(pad=0.8)
        fig.savefig(vis_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=110)
        plt.close(fig)
