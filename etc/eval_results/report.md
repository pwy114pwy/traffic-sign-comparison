# 交通标志检测模型性能对比报告

> 数据集：GTSRB（43类交通标志）
> 评估样本数：500 张图片
> 评估时间：2026-02-23 01:00:45

## 评估指标汇总

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 延迟(ms/img) | FPS |
|------|-----------|--------|---------|--------------|-------------|-----|
| YOLOv8 | 0.8317 | 0.9529 | 0.9822 | 0.8710 | 300.13 | 3.3 |
| RT-DETR | 0.9387 | 0.8984 | 0.9165 | 0.7736 | 1004.31 | 1.0 |

## 精度对比图

![精度对比](D:\traffic-sign-rtdetr\etc\eval_results\comparison_bar.png)


## 速度对比图

![速度对比](D:\traffic-sign-rtdetr\etc\eval_results\speed_comparison.png)


## 结论分析

- **精度最佳**：YOLOv8（mAP@0.5 = 0.9822）
- **速度最快**：YOLOv8（3.3 FPS）

### YOLOv8
- 权重路径：`D:\traffic-sign-rtdetr\runs\train\exp_yolov8\weights\best.pt`
- mAP@0.5 = **0.9822**，mAP@0.5:0.95 = 0.8710
- Precision = 0.8317，Recall = 0.9529
- 推理延迟 = 300.13 ms，吞吐量 = 3.3 FPS

### RT-DETR
- 权重路径：`D:\traffic-sign-rtdetr\runs\train\exp\weights\best.pt`
- mAP@0.5 = **0.9165**，mAP@0.5:0.95 = 0.7736
- Precision = 0.9387，Recall = 0.8984
- 推理延迟 = 1004.31 ms，吞吐量 = 1.0 FPS
