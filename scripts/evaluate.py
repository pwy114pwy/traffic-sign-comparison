"""
scripts/evaluate.py —— 双模型性能对比评估脚本

功能：
  1. 从 GTSRB 验证集随机采样一部分图片（默认 500 张）
  2. 创建临时子集 YAML，用 model.val() 分别评估 YOLOv8 和 RT-DETR
  3. 收集 mAP@0.5 / mAP@0.5:0.95 / Precision / Recall / 推理速度
  4. 生成对比条形图 + 速度对比图
  5. 输出 Markdown 格式评估报告

使用方法：
  python scripts/evaluate.py                  # 默认采样 500 张
  python scripts/evaluate.py --samples 200    # 采样 200 张（更快）
  python scripts/evaluate.py --full           # 使用完整验证集（慢）

结果保存至 etc/eval_results/
"""

import os
import sys
import time
import random
import shutil
import argparse
import platform
import pathlib
from pathlib import Path

# Windows 路径兼容
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO, RTDETR

# ============================================================
# 配置
# ============================================================
ROOT       = Path(__file__).resolve().parents[1]
DATA_YAML  = ROOT / "data" / "gtsrb.yaml"
OUTPUT_DIR = ROOT / "etc" / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型列表：(显示名称, 权重路径, 模型类)
MODELS = [
    ("YOLOv8",   str(ROOT / "runs" / "train" / "exp_yolov8" / "weights" / "best.pt"), YOLO),
    ("RT-DETR",  str(ROOT / "runs" / "train" / "exp"        / "weights" / "best.pt"), RTDETR),
]

IMG_SIZE   = 640
BATCH_SIZE = 16
DEVICE     = ""    # "" = 自动选 GPU/CPU

# ============================================================
# 采样子集
# ============================================================

def create_sampled_dataset(n_samples: int) -> str:
    """从验证集随机采样 n_samples 张图片，创建临时子数据集。

    返回新的 YAML 文件路径。
    """
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 获取完整 val 图片目录
    val_img_dir = (DATA_YAML.parent / cfg["val"]).resolve()
    if not val_img_dir.exists():
        print(f"❌ 验证集目录不存在: {val_img_dir}")
        sys.exit(1)

    # 收集所有图片
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs = [p for p in val_img_dir.rglob("*") if p.suffix.lower() in exts]
    total = len(all_imgs)
    print(f"  验证集共 {total} 张图片")

    if n_samples >= total:
        print(f"  采样数 >= 总数，使用完整验证集")
        return str(DATA_YAML)

    # 随机采样
    random.seed(42)  # 固定种子，保证可复现
    sampled = random.sample(all_imgs, n_samples)
    print(f"  随机采样 {n_samples} 张图片用于评估")

    # 创建临时目录
    tmp_dir = ROOT / "etc" / "_eval_tmp"
    tmp_img_dir = tmp_dir / "images" / "val"
    tmp_lbl_dir = tmp_dir / "labels" / "val"
    # 清理旧的临时数据
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_img_dir.mkdir(parents=True)
    tmp_lbl_dir.mkdir(parents=True)

    # 复制采样图片和对应标注
    for img_path in sampled:
        # 复制图片
        dst_img = tmp_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        # 复制对应标注（YOLO 格式：同名 .txt）
        lbl_name = img_path.stem + ".txt"
        # 标注路径：把 images 替换成 labels
        lbl_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, tmp_lbl_dir / lbl_name)

    # 写新的 YAML
    new_cfg = {
        "train": cfg.get("train", ""),
        "val":   str(tmp_img_dir),
        "nc":    cfg["nc"],
        "names": cfg["names"],
    }
    tmp_yaml = tmp_dir / "gtsrb_sample.yaml"
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"  临时数据集已创建: {tmp_yaml}")
    return str(tmp_yaml)


# ============================================================
# 评估
# ============================================================

def eval_model(name: str, weights: str, model_cls, data_yaml: str) -> dict:
    """加载并评估单个模型，返回指标字典。"""
    print(f"\n{'='*60}")
    print(f"► 正在评估: {name}")
    print(f"  权重: {weights}")
    print(f"{'='*60}")

    if not os.path.exists(weights):
        print(f"  ⚠️  权重文件不存在，跳过: {weights}")
        return None

    model = model_cls(weights)

    val_result = model.val(
        data    = data_yaml,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        verbose = False,
        plots   = False,
    )

    mp       = float(val_result.box.mp)    if val_result.box.mp   is not None else 0.0
    mr       = float(val_result.box.mr)    if val_result.box.mr   is not None else 0.0
    map50    = float(val_result.box.map50) if val_result.box.map50 is not None else 0.0
    map5095  = float(val_result.box.map)   if val_result.box.map   is not None else 0.0

    speed_ms = val_result.speed.get("inference", 0.0)

    metrics = {
        "name":       name,
        "weights":    weights,
        "precision":  mp,
        "recall":     mr,
        "map50":      map50,
        "map5095":    map5095,
        "speed_ms":   speed_ms,
        "fps":        1000.0 / speed_ms if speed_ms > 0 else 0.0,
    }

    print(f"  Precision : {mp:.4f}")
    print(f"  Recall    : {mr:.4f}")
    print(f"  mAP@0.5   : {map50:.4f}")
    print(f"  mAP@0.5:95: {map5095:.4f}")
    print(f"  推理延迟  : {speed_ms:.2f} ms/img  ({metrics['fps']:.1f} FPS)")
    return metrics


# ============================================================
# 图表
# ============================================================

def plot_bar_comparison(results: list):
    """生成各精度指标对比条形图。"""
    names        = [r["name"] for r in results]
    metrics_keys = ["precision", "recall", "map50", "map5095"]
    labels       = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
    model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    x      = np.arange(len(labels))
    width  = 0.35
    n      = len(results)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        vals = [res[k] for k in metrics_keys]
        bars = ax.bar(x + offsets[i], vals, width, label=res["name"],
                      color=model_colors[i % len(model_colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison — GTSRB", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "comparison_bar.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n📊 条形图已保存: {out}")
    return str(out)


def plot_speed_comparison(results: list):
    """生成速度对比图。"""
    names    = [r["name"] for r in results]
    speed_ms = [r["speed_ms"] for r in results]
    fps_vals = [r["fps"] for r in results]
    model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar(names, speed_ms,
                         color=[model_colors[i % len(model_colors)] for i in range(len(names))],
                         alpha=0.85)
    for bar, val in zip(bars1, speed_ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f} ms", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Latency (ms/img)", fontsize=11)
    axes[0].set_title("Inference Latency (lower is better)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    bars2 = axes[1].bar(names, fps_vals,
                         color=[model_colors[i % len(model_colors)] for i in range(len(names))],
                         alpha=0.85)
    for bar, val in zip(bars2, fps_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("FPS", fontsize=11)
    axes[1].set_title("Throughput FPS (higher is better)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Inference Speed Comparison — GTSRB", fontsize=14, fontweight="bold")
    out = OUTPUT_DIR / "speed_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"⚡ 速度对比图已保存: {out}")
    return str(out)


# ============================================================
# 报告
# ============================================================

def generate_report(results: list, bar_img: str, speed_img: str, n_samples: int):
    """生成 Markdown 格式评估报告。"""
    lines = []
    lines.append("# 交通标志检测模型性能对比报告\n")
    lines.append(f"> 数据集：GTSRB（43类交通标志）")
    lines.append(f"> 评估样本数：{n_samples} 张图片")
    lines.append(f"> 评估时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 评估指标汇总\n")
    lines.append("| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 延迟(ms/img) | FPS |")
    lines.append("|------|-----------|--------|---------|--------------|-------------|-----|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['map50']:.4f} | {r['map5095']:.4f} | "
            f"{r['speed_ms']:.2f} | {r['fps']:.1f} |"
        )

    lines.append(f"\n## 精度对比图\n")
    lines.append(f"![精度对比]({bar_img})\n")
    lines.append(f"\n## 速度对比图\n")
    lines.append(f"![速度对比]({speed_img})\n")

    lines.append("\n## 结论分析\n")
    if len(results) >= 2:
        best_map = max(results, key=lambda x: x["map50"])
        best_spd = max(results, key=lambda x: x["fps"])
        lines.append(f"- **精度最佳**：{best_map['name']}（mAP@0.5 = {best_map['map50']:.4f}）")
        lines.append(f"- **速度最快**：{best_spd['name']}（{best_spd['fps']:.1f} FPS）\n")

        for r in results:
            lines.append(f"### {r['name']}")
            lines.append(f"- 权重路径：`{r['weights']}`")
            lines.append(f"- mAP@0.5 = **{r['map50']:.4f}**，mAP@0.5:0.95 = {r['map5095']:.4f}")
            lines.append(f"- Precision = {r['precision']:.4f}，Recall = {r['recall']:.4f}")
            lines.append(f"- 推理延迟 = {r['speed_ms']:.2f} ms，吞吐量 = {r['fps']:.1f} FPS\n")

    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 评估报告已保存: {report_path}")
    return str(report_path)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="双模型性能对比评估")
    parser.add_argument("--samples", type=int, default=500,
                        help="从验证集随机采样的图片数量（默认 500）")
    parser.add_argument("--full", action="store_true",
                        help="使用完整验证集（不采样，较慢）")
    args = parser.parse_args()

    print("🚀 开始模型性能评估...\n")
    print(f"  原始数据配置: {DATA_YAML}")
    print(f"  输出目录: {OUTPUT_DIR}")

    # 创建采样子集或使用完整集
    if args.full:
        data_yaml = str(DATA_YAML)
        n_samples = "全部"
        print("  模式：完整验证集评估")
    else:
        data_yaml = create_sampled_dataset(args.samples)
        n_samples = args.samples

    # 评估每个模型
    results = []
    for name, weights, model_cls in MODELS:
        res = eval_model(name, weights, model_cls, data_yaml)
        if res:
            results.append(res)

    if not results:
        print("\n❌ 没有可用的模型权重，请先训练模型。")
        return

    # 生成图表和报告
    bar_img   = plot_bar_comparison(results)
    speed_img = plot_speed_comparison(results)
    report    = generate_report(results, bar_img, speed_img, n_samples)

    # 清理临时目录
    tmp_dir = ROOT / "etc" / "_eval_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print("🧹 临时数据已清理")

    print(f"\n✅ 评估完成！结果目录: {OUTPUT_DIR}")
    print(f"   报告: {report}")


if __name__ == "__main__":
    main()
