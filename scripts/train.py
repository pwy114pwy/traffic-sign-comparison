# scripts/train.py —— RT-DETR 交通标志检测训练脚本
"""
使用方法:
    python scripts/train.py

训练完成后，权重保存在 runs/detect/train/weights/best.pt
将 .env 中的 WEIGHTS_PATH 修改为该路径后重启 Web 服务即可。
"""
from pathlib import Path
from ultralytics import RTDETR

# ============================================================
# 训练配置（按需修改）
# ============================================================
MODEL_SIZE  = "rtdetr-l"      # 可选: rtdetr-l / rtdetr-x
DATA_YAML   = str(Path(__file__).parents[1] / "data" / "gtsrb.yaml")
EPOCHS      = 100             # 建议 100 以上
IMG_SIZE    = 640
BATCH_SIZE  = 8               # 根据显存调整（8G 显存建议 8，16G 建议 16）
DEVICE      = 0               # GPU 编号，CPU 训练设为 'cpu'
WORKERS     = 4
PROJECT     = "runs/detect"
NAME        = "rtdetr_gtsrb"
# ============================================================

def main():
    model = RTDETR(f"{MODEL_SIZE}.pt")   # 自动下载官方预训练权重

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        device    = DEVICE,
        workers   = WORKERS,
        project   = PROJECT,
        name      = NAME,
        # RT-DETR 推荐超参数
        lr0       = 1e-4,
        lrf       = 0.01,
        warmup_epochs = 3,
        patience  = 30,        # 早停耐心值
        save      = True,
        verbose   = True,
    )
    print(f"\n✅ 训练完成！最佳权重: {results.save_dir}/weights/best.pt")
    print(f"请将 .env 中的 WEIGHTS_PATH 修改为: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
