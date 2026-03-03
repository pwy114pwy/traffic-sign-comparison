# web/app.py  ——  RT-DETR 交通标志检测 Web 服务
import os
import cv2
import time
import logging
import requests
import numpy as np
from io import BytesIO
from pathlib import Path
from flask import Flask, request, render_template, jsonify
import dotenv
from ultralytics import YOLO, RTDETR

# ============================================================
# 环境变量
# ============================================================
dotenv.load_dotenv()

# ============================================================
# 日志
# ============================================================
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file  = os.getenv("LOG_FILE", "logs/app.log")
os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================
FILE         = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]

WEIGHTS_PATH    = os.getenv("WEIGHTS_PATH", "rtdetr-l.pt")
MODEL_TYPE      = os.getenv("MODEL_TYPE", "auto").lower()  # yolo / rtdetr / auto
UPLOAD_FOLDER   = os.getenv("UPLOAD_FOLDER", str(PROJECT_ROOT / "web" / "static" / "uploads"))
RESULT_FOLDER   = os.getenv("RESULT_FOLDER", str(PROJECT_ROOT / "web" / "static" / "results"))
ANOMALY_CONF    = float(os.getenv("ANOMALY_CONF_THRES", "0.5"))
SCALE_THRESHOLD = int(os.getenv("SCALE_THRESHOLD", "200"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

logger.info(f"📦 WEIGHTS_PATH : {WEIGHTS_PATH}")
logger.info(f"📤 UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"📥 RESULT_FOLDER: {RESULT_FOLDER}")

# ============================================================
# 模型加载  (Ultralytics RT-DETR —— 一行搞定)
# ============================================================
def _load_model(weights: str, model_type: str):
    """根据 MODEL_TYPE 或权重文件名自动选择模型类"""
    if model_type == "rtdetr":
        return RTDETR(weights)
    if model_type == "yolo":
        return YOLO(weights)
    # auto: 文件名含 rtdetr 则用 RTDETR，否则用 YOLO
    if "rtdetr" in weights.lower():
        logger.info("🔍 AUTO 检测到 RT-DETR 权重，使用 RTDETR 加载")
        return RTDETR(weights)
    logger.info("🔍 AUTO 检测到 YOLO 权重，使用 YOLO 加载")
    return YOLO(weights)

try:
    logger.info(f"⏳ 正在加载模型 [{MODEL_TYPE.upper()}]: {WEIGHTS_PATH}")
    model = _load_model(WEIGHTS_PATH, MODEL_TYPE)
    # 预热推理，减少第一次推理延迟
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    logger.info(f"✅ 模型加载成功: {type(model).__name__}")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
    raise SystemExit(1)

app = Flask(__name__)

# ============================================================
# 辅助函数
# ============================================================

def maybe_upscale(img: np.ndarray):
    """若图片过小则等比放大，返回 (新图, scale_factor)"""
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < SCALE_THRESHOLD:
        factor = SCALE_THRESHOLD / min_dim
        img = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LINEAR)
        return img, factor
    return img, 1.0


def run_detection(img0: np.ndarray, conf_thres: float, iou_thres: float):
    """
    对 BGR numpy 图像执行推理，返回：
      - annotated_img : 已标注的 BGR 图像（可能经过放大，仅用于显示）
      - stats         : 统计字典

    注意：模型始终在原始尺寸上推理，upscale 只影响结果图像显示。
    """
    t0 = time.time()
    results = model.predict(
        img0,           # 输入原始 BGR 图像（通常是 cv2 读取的格式）。
        conf=conf_thres,# 置信度阈值：模型对自己判断把握低于此值的框会被过滤掉。
        iou=iou_thres,  # 交并比阈值：用于非极大值抑制（NMS），消除重叠过高的重复框。
        verbose=False,  # 设置为 False，不在终端打印推理细节（如“1 person, 2 cars”）。
        save=False,     # 不自动将预测后的图片保存到硬盘，节省磁盘 IO 耗时。
    )
    inference_time = time.time() - t0

    # 取第一张结果（单张图推理）
    result = results[0]

    # 若图片过小则放大——仅用于显示，不影响推理结果
    # line_width=1 / font_size=8：避免小图放大后线条过粗、标签溢出
    h, w = img0.shape[:2]
    lw = max(1, min(2, round(min(h, w) / 320)))
    display_img, _ = maybe_upscale(result.plot(line_width=lw, font_size=8))

    # 统计
    boxes      = result.boxes  # 获取所有的检测框对象。
    class_ids  = boxes.cls.cpu().numpy().astype(int) if boxes is not None else []
    confs      = boxes.conf.cpu().numpy()             if boxes is not None else []
    names_list = result.names   # dict: id -> name

    detection_count    = len(class_ids)
    class_counts: dict = {}
    total_conf         = 0.0
    anomaly_count      = 0

    for cls_id, conf in zip(class_ids, confs):
        name = names_list[cls_id]
        class_counts[name] = class_counts.get(name, 0) + 1
        total_conf += float(conf)
        if float(conf) < ANOMALY_CONF:
            anomaly_count += 1

    detections = sorted(
        [{'name': n, 'count': c,
          'percentage': (c / detection_count * 100) if detection_count else 0.0}
         for n, c in class_counts.items()],
        key=lambda x: x['count'], reverse=True
    )

    avg_confidence = (total_conf / detection_count * 100) if detection_count else 0.0

    stats = dict(
        detection_count    = detection_count,
        unique_classes_count = len(class_counts),
        avg_confidence     = avg_confidence,
        anomaly_count      = anomaly_count,
        inference_time     = inference_time,
        class_counts       = class_counts,
        detections         = detections,
    )
    return display_img, stats


def download_image(url: str) -> np.ndarray:
    """从 URL 下载图片，返回 BGR ndarray（含代理/SSL 容错）"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, timeout=15, headers=headers, verify=False)
        resp.raise_for_status()
    except requests.exceptions.ProxyError:
        resp = requests.get(url, timeout=15, headers=headers,
                            verify=False, proxies={'http': None, 'https': None})
        resp.raise_for_status()

    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片，请确认链接指向有效的图片文件")
    return img


def safe_filename(url_or_name: str, fallback: str = "image.jpg") -> str:
    """从 URL 或文件名中提取合法图片文件名"""
    from urllib.parse import urlparse
    name = os.path.basename(urlparse(url_or_name).path) or fallback
    if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        name += '.jpg'
    return name


# ============================================================
# 路由
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """本地图片文件检测"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未上传文件'}), 400
        file = request.files['image']
        if not file or not file.filename:
            return jsonify({'error': '文件为空'}), 400

        allowed = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        if not file.filename.lower().endswith(allowed):
            return jsonify({'error': f'不支持的格式，请上传 {", ".join(allowed)} 图片'}), 400

        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres  = float(request.form.get('iou_thres',  0.45))

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        img0 = cv2.imread(input_path)
        if img0 is None:
            os.remove(input_path)
            return jsonify({'error': '无法读取图片，文件可能已损坏'}), 400

        annotated_img, stats = run_detection(img0, conf_thres, iou_thres)

        output_path = os.path.join(RESULT_FOLDER, file.filename)
        cv2.imwrite(output_path, annotated_img)
        logger.info(f"图片检测完成: {stats['detection_count']} 个目标, 耗时 {stats['inference_time']:.3f}s")

        return render_template('result.html',
                               img_path=file.filename,
                               conf_thres=conf_thres,
                               iou_thres=iou_thres,
                               **stats)

    except Exception as e:
        logger.error(f"图片检测失败: {e}", exc_info=True)
        return jsonify({'error': f'检测失败: {e}'}), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """通过图片 URL 检测"""
    try:
        image_url = request.form.get('image_url', '').strip()
        if not image_url:
            return jsonify({'error': '未提供图片网址'}), 400
        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': '请输入有效的 http/https 图片网址'}), 400

        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres  = float(request.form.get('iou_thres',  0.45))

        logger.info(f"正在下载图片: {image_url}")
        try:
            img0 = download_image(image_url)
        except requests.exceptions.Timeout:
            return jsonify({'error': '下载图片超时，请检查网址或网络'}), 400
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'下载图片失败: {e}'}), 400
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        filename = safe_filename(image_url)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, filename), img0)

        annotated_img, stats = run_detection(img0, conf_thres, iou_thres)

        output_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(output_path, annotated_img)
        logger.info(f"URL 图片检测完成: {stats['detection_count']} 个目标")

        return render_template('result.html',
                               img_path=filename,
                               conf_thres=conf_thres,
                               iou_thres=iou_thres,
                               **stats)

    except Exception as e:
        logger.error(f"URL 图片检测失败: {e}", exc_info=True)
        return jsonify({'error': f'检测失败: {e}'}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """视频文件检测"""
    cap = out = None
    try:
        if 'video' not in request.files:
            return jsonify({'error': '未上传视频文件'}), 400
        file = request.files['video']
        if not file or not file.filename:
            return jsonify({'error': '视频文件为空'}), 400

        allowed = ('.mp4', '.avi', '.mov', '.wmv', '.mkv')
        if not file.filename.lower().endswith(allowed):
            return jsonify({'error': f'不支持的格式，请上传 {", ".join(allowed)} 视频'}), 400

        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres  = float(request.form.get('iou_thres',  0.45))

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            os.remove(input_path)
            return jsonify({'error': '无法打开视频文件，文件可能已损坏'}), 400

        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        output_path = os.path.join(RESULT_FOLDER, file.filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        detection_count = 0
        class_counts: dict = {}
        total_conf      = 0.0
        anomaly_count   = 0
        processed       = 0
        t0              = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed += 1

                results = model.predict(frame, conf=conf_thres, iou=iou_thres,
                                        verbose=False, save=False)
                result  = results[0]
                annotated = result.plot()

                boxes     = result.boxes
                cls_ids   = boxes.cls.cpu().numpy().astype(int) if boxes else []
                confs_arr = boxes.conf.cpu().numpy()             if boxes else []
                names_map = result.names

                frame_count = len(cls_ids)
                detection_count += frame_count

                for cls_id, conf in zip(cls_ids, confs_arr):
                    name = names_map[cls_id]
                    class_counts[name] = class_counts.get(name, 0) + 1
                    total_conf += float(conf)
                    if float(conf) < ANOMALY_CONF:
                        anomaly_count += 1

                # FPS 叠加
                elapsed = time.time() - t0
                realtime_fps = processed / elapsed if elapsed > 0 else 0
                cv2.putText(annotated, f'FPS: {realtime_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated, f'Det: {frame_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(annotated)

                if processed % 100 == 0:
                    logger.info(f"已处理 {processed}/{total_frames} 帧")
        finally:
            cap.release()
            out.release()

        total_time = time.time() - t0
        avg_fps    = processed / total_time if total_time > 0 else 0
        avg_confidence = (total_conf / detection_count * 100) if detection_count else 0.0

        detections = sorted(
            [{'name': n, 'count': c,
              'percentage': (c / detection_count * 100) if detection_count else 0.0}
             for n, c in class_counts.items()],
            key=lambda x: x['count'], reverse=True
        )

        logger.info(f"视频处理完成: {processed}帧, 耗时 {total_time:.2f}s, FPS {avg_fps:.2f}")

        return render_template('result.html',
                               video_path=file.filename,
                               detection_count=detection_count,
                               unique_classes_count=len(class_counts),
                               avg_confidence=avg_confidence,
                               anomaly_count=anomaly_count,
                               avg_fps=avg_fps,
                               total_time=total_time,
                               detections=detections,
                               class_counts=class_counts)

    except Exception as e:
        logger.error(f"视频检测失败: {e}", exc_info=True)
        if cap:  cap.release()
        if out:  out.release()
        return jsonify({'error': f'视频检测失败: {e}'}), 500


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    flask_host  = os.getenv("FLASK_HOST",  "0.0.0.0")
    flask_port  = int(os.getenv("FLASK_PORT",  "5000"))
    flask_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"🚀 启动 RT-DETR Web 服务: {flask_host}:{flask_port}")
    app.run(host=flask_host, port=flask_port, debug=flask_debug)
