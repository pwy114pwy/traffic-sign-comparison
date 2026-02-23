"""
scripts/compare_images.py —— 双模型推理可视化对比脚本

功能：
  从 GTSRB 验证集随机抽取 N 张图片，分别用 YOLOv8 和 RT-DETR 推理，
  生成左右并排的对比网格图，保存到 etc/eval_results/compare_grid.png。

使用方法：
  python scripts/compare_images.py              # 默认抽 6 张
  python scripts/compare_images.py --samples 4  # 指定抽 4 张
  python scripts/compare_images.py --image path/to/img.jpg  # 指定单张图片
"""

import argparse
import os
import platform
import pathlib
import random
from pathlib import Path

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO, RTDETR


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ============================================================
# 配置
# ============================================================
ROOT       = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "etc" / "eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 两个模型
MODEL_A_NAME    = "YOLOv8（fine-tuned）"
MODEL_A_WEIGHTS = str(ROOT / "runs" / "train" / "exp_yolov8" / "weights" / "best.pt")
MODEL_A_CLS     = YOLO

MODEL_B_NAME    = "RT-DETR（fine-tuned）"
MODEL_B_WEIGHTS = str(ROOT / "runs" / "train" / "exp" / "weights" / "best.pt")
MODEL_B_CLS     = RTDETR

# GTSRB 图片目录（项目内部，无独立验证集时用 train）
VAL_DIR = ROOT / "datasets" / "GTSRB" / "images" / "train"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# 工具函数
# ============================================================

def load_model(weights: str, model_cls):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")
    print(f"  ► 加载 {model_cls.__name__}: {weights}")
    return model_cls(weights)


def predict_and_plot(model, img_bgr: np.ndarray, line_width: int = 1) -> np.ndarray:
    """推理单张图片，返回带标注的 BGR numpy 图。"""
    results = model.predict(img_bgr, verbose=False, save=False, conf=0.25, iou=0.45)
    return results[0].plot(line_width=line_width, font_size=8)


def collect_val_images(val_dir: Path, n: int) -> list:
    """从验证集目录随机采样 n 张图片路径。"""
    imgs = [p for p in val_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"验证集目录为空或不存在: {val_dir}")
    random.shuffle(imgs)
    return [str(p) for p in imgs[:n]]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# 主流程
# ============================================================

def run(image_paths: list, output_name: str = "compare_grid.png"):
    print("\n🚀 开始加载模型...\n")
    model_a = load_model(MODEL_A_WEIGHTS, MODEL_A_CLS)
    model_b = load_model(MODEL_B_WEIGHTS, MODEL_B_CLS)

    n = len(image_paths)
    print(f"\n📸 共 {n} 张图片，开始推理对比...\n")

    # 3 列：原图 | 模型A | 模型B
    ncols = 3
    nrows = n
    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.05, wspace=0.05)

    col_titles = ["原图", MODEL_A_NAME, MODEL_B_NAME]

    for row, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ⚠️ 无法读取图片: {img_path}")
            continue

        ann_a = predict_and_plot(model_a, img_bgr)
        ann_b = predict_and_plot(model_b, img_bgr)

        imgs_row = [bgr_to_rgb(img_bgr), bgr_to_rgb(ann_a), bgr_to_rgb(ann_b)]

        for col, img_show in enumerate(imgs_row):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_show)
            ax.axis("off")
            # 第一行加列标题
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13,
                             fontweight="bold", pad=8)
            # 每行最左侧加图片文件名
            if col == 0:
                fname = Path(img_path).name
                ax.set_ylabel(fname, fontsize=8, rotation=0,
                              labelpad=60, va="center")

        print(f"  ✓ [{row+1}/{n}] {Path(img_path).name}")

    fig.suptitle("双模型推理结果可视化对比", fontsize=16,
                 fontweight="bold", y=1.01)

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 对比图已保存: {out_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="双模型推理可视化对比")
    parser.add_argument("--samples", type=int, default=6,
                        help="从验证集随机抽取的图片数量（默认 6）")
    parser.add_argument("--image", type=str, default=None,
                        help="指定单张图片路径（指定后忽略 --samples）")
    parser.add_argument("--output", type=str, default="compare_grid.png",
                        help="输出文件名（默认 compare_grid.png）")
    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 图片不存在: {args.image}")
            return
        image_paths = [args.image]
    else:
        print(f"🔍 从验证集采样 {args.samples} 张图片: {VAL_DIR}")
        try:
            image_paths = collect_val_images(VAL_DIR, args.samples)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("请确认 GTSRB 验证集已下载到 ../datasets/gtsrb/images/val/")
            return

    run(image_paths, output_name=args.output)


if __name__ == "__main__":
    main()
