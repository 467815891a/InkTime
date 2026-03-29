#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import base64
import json
import sqlite3
import os
import subprocess
import time
import requests
import io
from PIL import Image, ExifTags, ImageOps
import config as cfg
import shutil


# =======================
# NAS 掉盘守护（macOS /Volumes）
# =======================
NAS_MOUNT_URL = str(getattr(cfg, "NAS_MOUNT_URL", "") or "").strip()
NAS_MOUNT_POINT = Path(str(getattr(cfg, "NAS_MOUNT_POINT", "/Volumes/photo") or "/Volumes/photo")).expanduser()
NAS_RETRY_TIMES = int(getattr(cfg, "NAS_RETRY_TIMES", 3) or 3)
NAS_RETRY_SLEEP_SEC = float(getattr(cfg, "NAS_RETRY_SLEEP_SEC", 2.0) or 2.0)


def _is_mount_ok() -> bool:
    """判断 NAS 是否仍然挂载（尽量保守）。"""
    try:
        # /Volumes 下的网络卷一般是 mount point
        if NAS_MOUNT_POINT and NAS_MOUNT_POINT.exists():
            if os.path.ismount(str(NAS_MOUNT_POINT)):
                return True
        # 兜底：只要有一个图片目录可访问也算 OK
        return any(img_dir.exists() for img_dir in IMAGE_DIRS)
    except Exception:
        return False


def _try_remount_nas() -> bool:
    """尝试重挂载 NAS。

    只在配置了 NAS_MOUNT_URL 时执行；优先使用 macOS 的 AppleScript 挂载，
    这样可以复用钥匙串里保存的凭据。
    """
    if not NAS_MOUNT_URL:
        return False

    print(f"[WARN] 检测到 NAS 可能掉盘，尝试重挂载：{NAS_MOUNT_URL}")

    # 1) AppleScript（推荐）：mount volume "afp://..." / "smb://..."
    try:
        subprocess.run(
            ["osascript", "-e", f'mount volume "{NAS_MOUNT_URL}"'],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        pass

    # 2) 兜底：如果你更喜欢命令行挂载，可以自行改成 mount_afp / mount_smbfs。

    # 等待卷出现
    for _ in range(10):
        if _is_mount_ok():
            print("[INFO] NAS 重挂载成功，继续处理。")
            return True
        time.sleep(0.5)

    print("[WARN] NAS 重挂载失败（卷仍不可用）。")
    return False


def _read_bytes_with_nas_retry(path: Path) -> bytes:
    """读文件：遇到 NAS 掉盘类错误时，尝试重挂载并重试。"""
    last_err: Exception | None = None

    # 只对照片库路径内的文件启用重挂载逻辑，避免误伤本地文件。
    try:
        in_photo_dir = any(str(path).startswith(str(img_dir)) for img_dir in IMAGE_DIRS)
    except Exception:
        in_photo_dir = False

    for attempt in range(1, max(1, NAS_RETRY_TIMES) + 1):
        try:
            return path.read_bytes()
        except OSError as e:
            last_err = e

            # 不在照片库目录内，或者没有配置 NAS URL，直接抛
            if not in_photo_dir or not NAS_MOUNT_URL:
                raise

            # 常见网络卷断连错误：57 Socket is not connected；也可能表现为 5/6 等 I/O 类错误。
            # 这里不做过度精确匹配，先判断挂载状态；如果掉盘就尝试重挂载。
            if not _is_mount_ok():
                print(f"[WARN] 读文件失败（第 {attempt}/{NAS_RETRY_TIMES} 次）：{e}")
                ok = _try_remount_nas()
                if ok:
                    # 重挂载后立刻重试
                    continue

            # 如果挂载看起来还 OK，但仍然读失败，按重试策略稍等再试
            if attempt < NAS_RETRY_TIMES:
                print(f"[WARN] 读文件失败（第 {attempt}/{NAS_RETRY_TIMES} 次），稍后重试：{e}")
                time.sleep(max(0.1, NAS_RETRY_SLEEP_SEC))
                continue

            raise

    # 理论上不会到这
    if last_err:
        raise last_err
    raise OSError("读取文件失败")


# ================== 配置区域（来自 config.py） ==================

ROOT_DIR = Path(__file__).resolve().parent

# 要扫描的图片目录
# 将IMAGE_DIR改为路径列表
image_dir_config = getattr(cfg, "IMAGE_DIR", "")
if isinstance(image_dir_config, str):
    # 支持逗号分隔的多个路径
    image_dirs = [d.strip() for d in image_dir_config.split(",") if d.strip()]
else:
    # 如果已经是列表，直接使用
    image_dirs = list(image_dir_config) if image_dir_config else []

# 处理每个路径
IMAGE_DIRS = []
for img_dir in image_dirs:
    path = Path(str(img_dir)).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    IMAGE_DIRS.append(path)

# 为了向后兼容，保持IMAGE_DIR作为第一个路径
IMAGE_DIR = IMAGE_DIRS[0] if IMAGE_DIRS else Path("").resolve()

# SQLite 数据库路径
DB_PATH = Path(str(getattr(cfg, "DB_PATH", "photos.db") or "photos.db")).expanduser()
if not DB_PATH.is_absolute():
    DB_PATH = (ROOT_DIR / DB_PATH).resolve()

# LM Studio/OpenAI 兼容接口（仍允许用环境变量覆盖）
API_URL = str(
    getattr(cfg, "API_URL", None)
    or os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
)

# 模型名称（仍允许用环境变量覆盖）
MODEL_NAME = str(
    getattr(cfg, "MODEL_NAME", None)
    or os.environ.get("LMSTUDIO_MODEL", "qwen3-vl-32b-instruct")
)

# API KEY（仍允许用环境变量覆盖）
API_KEY = str(getattr(cfg, "API_KEY", None) or os.environ.get("LMSTUDIO_API_KEY", ""))

# 每次处理多少张；None 为不限制
BATCH_LIMIT = getattr(cfg, "BATCH_LIMIT", None)

# 请求超时时间（秒）
TIMEOUT = float(getattr(cfg, "TIMEOUT", 600) or 600)

# 发送给 VLM 之前，先把图片长边缩放到该值（像素）。
# 0 表示不缩放。
# 本地推理可保持较高值；云端推理建议降低（减少 token/成本）。
VLM_MAX_LONG_EDGE = int(getattr(cfg, "VLM_MAX_LONG_EDGE", 2560) or 2560)

# 中文城市数据库位置
WORLD_CITIES_CSV = Path(str(getattr(cfg, "WORLD_CITIES_CSV", "data/world_cities_zh.csv") or "data/world_cities_zh.csv")).expanduser()
if not WORLD_CITIES_CSV.is_absolute():
    WORLD_CITIES_CSV = (ROOT_DIR / WORLD_CITIES_CSV).resolve()

CITY_GRID_DEG = float(getattr(cfg, "CITY_GRID_DEG", 1.0) or 1.0)
CITY_MAX_DISTANCE_KM = float(getattr(cfg, "CITY_MAX_DISTANCE_KM", 80.0) or 80.0)
HOME_LAT = float(getattr(cfg, "HOME_LAT", 22.543096) or 22.543096)
HOME_LON = float(getattr(cfg, "HOME_LON", 114.057865) or 114.057865)
HOME_RADIUS_KM = float(getattr(cfg, "HOME_RADIUS_KM", 60.0) or 60.0)
# ==================================================

# exiftool 是否可用：缺失时只降级 GPS/部分 EXIF，不中断流程
EXIFTOOL_AVAILABLE = False

def require_exiftool() -> None:
    global EXIFTOOL_AVAILABLE
    EXIFTOOL_AVAILABLE = shutil.which("exiftool") is not None
    if not EXIFTOOL_AVAILABLE:
        print(
            "[WARN] 未找到 exiftool，将跳过 exiftool 辅助的 GPS/EXIF 读取（不影响主流程）。\n"
            "       如需更完整的 GPS 信息，请安装：\n"
            "       macOS: brew install exiftool\n"
            "       Ubuntu/Debian: sudo apt-get install -y libimage-exiftool-perl\n"
            "       Windows: choco install exiftool"
        )

def encode_image_to_b64(path: Path) -> str:
    """读取图片并（可选）缩放长边后，重新编码为 JPEG，再转 base64。

    目的：
    1) 控制输入分辨率（尤其是 200MP 这类超大图），避免推理成本/延迟暴涨；
    2) 通过重新编码，尽量规避某些 JPEG 在 libvips 侧解码报错（如 marker 前多余字节）。
    """
    data = _read_bytes_with_nas_retry(path)

    # 尽量用 PIL 容错打开，然后统一转成干净的 JPEG bytes
    try:
        img = Image.open(io.BytesIO(data))
        # 处理 EXIF 旋转
        try:
            img = ImageOps.exif_transpose(img)  # type: ignore
        except Exception:
            pass

        # 统一色彩模式：JPEG 需要 RGB
        if img.mode in ("RGBA", "LA"):
            # 有透明通道时，用白底合成
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 可选缩放
        try:
            w, h = img.size
            long_edge = max(w, h)
            if VLM_MAX_LONG_EDGE and long_edge > VLM_MAX_LONG_EDGE:
                scale = float(VLM_MAX_LONG_EDGE) / float(long_edge)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        except Exception:
            pass

        out = io.BytesIO()
        # quality 92 在观感和体积之间比较平衡；optimize 可能更慢但通常能降体积
        img.save(out, format="JPEG", quality=92, optimize=True)
        clean_bytes = out.getvalue()
        return base64.b64encode(clean_bytes).decode("utf-8")

    except Exception:
        # 兜底：如果 PIL 也打不开，就退回原始 bytes（让上游报错更直观）
        return base64.b64encode(data).decode("utf-8")


def ensure_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS photo_scores (
            path              TEXT PRIMARY KEY,
            caption           TEXT,
            type              TEXT,
            memory_score      REAL,
            beauty_score      REAL,
            reason            TEXT,
            width             INTEGER,
            height            INTEGER,
            orientation       TEXT,
            used_at           TEXT,
            exif_json         TEXT,
            raw_json          TEXT,
            exif_datetime     TEXT,
            exif_make         TEXT,
            exif_model        TEXT,
            exif_iso          INTEGER,
            exif_exposure_time REAL,
            exif_f_number     REAL,
            exif_focal_length REAL,
            exif_gps_lat      REAL,
            exif_gps_lon      REAL,
            exif_gps_alt      REAL,
            side_caption      TEXT,
            exif_city         TEXT
        )
        """
    )
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_json TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN width INTEGER")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN height INTEGER")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN orientation TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN used_at TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_datetime TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_make TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_model TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_iso INTEGER")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_exposure_time REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_f_number REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_focal_length REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_gps_lat REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_gps_lon REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_gps_alt REAL")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN side_caption TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE photo_scores ADD COLUMN exif_city TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()

# 从 Thinking 模型的输出中提取最终答案
def extract_final_answer(content: str) -> str:
    """
    处理 Thinking 模型的输出格式，提取最终答案。
    
    Thinking 模型可能会输出思考过程（如 <thinking>...</thinking>），
    需要将其剥离，只保留最终答案。
    """
    if not content:
        return ""
    
    import re
    
    # 步骤 1：去除 XML 风格的思考过程标签
    result = content
    
    # 去除 <thinking>...</thinking>
    result = re.sub(r'<thinking>.*?</thinking>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除 <thought>...</thought>
    result = re.sub(r'<thought>.*?</thought>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除 <reasoning>...</reasoning>
    result = re.sub(r'<reasoning>.*?</reasoning>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除 <reason>...</reason>
    result = re.sub(r'<reason>.*?</reason>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除 <analysis>...</analysis>
    result = re.sub(r'<analysis>.*?</analysis>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 去除 <step>...</step>
    result = re.sub(r'<step>.*?</step>', '', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 步骤 2：去除常见的思考过程前缀
    prefixes_to_remove = [
        "让我思考一下",
        "让我来思考",
        "我来思考一下",
        "思考：",
        "分析：",
        "Reasoning:",
        "Thought:",
        "Thinking:",
        "好的，我来生成一句文案",
        "根据图片内容",
        "这张照片展示",
        "画面中",
        "我看到",
        "从照片中",
        "基于图片",
        "让我分析一下",
        "我来分析",
        "首先",
        "然后",
        "最后",
        "综上所述",
    ]
    
    cleaned_result = result.strip()
    for prefix in prefixes_to_remove:
        # 使用正则匹配，处理大小写和标点
        pattern = re.escape(prefix) + r'[:：]?\s*'
        cleaned_result = re.sub(pattern, '', cleaned_result, flags=re.IGNORECASE)
    
    # 步骤 3：如果内容仍然很长，可能包含多行思考过程
    lines = cleaned_result.split('\n')
    lines = [line.strip() for line in lines if line.strip()]  # 去除空行
    
    if len(lines) > 3:
        # 尝试找到最后一行或最后几行作为最终答案
        # 通常最终答案在最后，且长度合理（5-50个字符）
        for i in range(len(lines)):
            line = lines[i]
            if 5 <= len(line) <= 50:
                # 如果这行看起来像最终答案（不包含思考关键词）
                if not any(keyword in line.lower() for keyword in ['思考', '分析', 'reason', 'think', '首先', '然后', '最后']):
                    # 从这一行开始取剩余内容
                    potential_answer = '\n'.join(lines[i:])
                    if len(potential_answer) <= 100:  # 总长度不超过100
                        cleaned_result = potential_answer
                        break
    
    # 步骤 4：去除引用符号和多余空白
    cleaned_result = cleaned_result.strip()
    cleaned_result = cleaned_result.strip('"""\'\'『』「」【】《》')
    
    # 步骤 5：如果结果仍然为空或过长，尝试其他策略
    if not cleaned_result or len(cleaned_result) > 100:
        # 策略：取最后一行
        if lines:
            last_line = lines[-1].strip()
            if 5 <= len(last_line) <= 50:
                cleaned_result = last_line
    
    return cleaned_result

# 生成一句话文案
def generate_side_caption(image_path: Path) -> str | None:
    system_prompt = (
        "你是一位为「电子相框」撰写旁白短句的中文文案助手。\n"
        "你的目标不是描述画面，而是为画面补上一点“画外之意”。\n"
        "创作原则：\n"
        "1. 只基于图片中能确定的信息进行联想，不要虚构时间、人物关系、事件背景。\n"
        "2. 文案应自然、有趣，带一点幽默或者诗意。\n"
        "3. 不要复述画面内容本身，而是写“看完画面后，心里多出来的一句话”。\n"
        "4. 相信你的第一直觉，联想不超过3个候选文案，最后按照第5原则选出最佳的。\n"
        "5. 可以从以下风格中思考：\n"
        "   - 让人感觉温情、治愈、平静\n"
        "   - 日常中的微妙情绪\n"
        "   - 对时间、记忆、瞬间的含蓄感受\n"
        "   - 看似平淡但有余味的一句判断\n"
        "格式要求：\n"
        "1. 只输出一句中文短句，不要换行，不要引号，不要任何解释。\n"
        "2. 长度最多不超过 30 个汉字。\n"
        "3. 不要出现“这张照片”“这一刻”“那天”等指代照片本身的词。\n"
    )
    user_prompt = "请基于这张照片，生成一句符合规则的中文文案。"
    try:
        img_b64 = encode_image_to_b64(image_path)
    except Exception:
        return None

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
        "repeat_penalty": 1,
        "top_p": 0.6,
        "top_k": 10,
        "typical_p": 1.0,
        "stream": False,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=max(480, TIMEOUT))
        print(f"[DEBUG] 响应状态码: {resp.status_code}")
    except Exception as e:
        print(f"[DEBUG] 请求失败: {e}")
        return None

    if not resp.ok:
        print(f"[DEBUG] HTTP 错误: {resp.status_code}")
        return None

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"[DEBUG] 1. 原始响应内容: '{data}'")
    except Exception as e:
        print(f"[DEBUG] 解析失败: {e}")
        return None

    if not isinstance(content, str):
        content = str(content)

    # 提取最终答案（处理 Thinking 模型的思考过程输出）
    final_content = extract_final_answer(content)
    caption = final_content.strip().strip("“”\"'")
    
    return caption or None


def list_images(limit: int | None = None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".HEIC"}
    files = []
    print("[INFO] 正在递归扫描图片目录，请稍候……")
    scanned = 0
    for img_dir in IMAGE_DIRS:
        print(f"[INFO] 扫描目录: {img_dir}")
        for p in img_dir.rglob("*"):
            scanned += 1
            if scanned % 500 == 0:
                print(f"[SCAN] 已扫描文件数：{scanned} …")
            if p.is_file() and p.suffix.lower() in exts:
                if is_screenshot(p):
                    continue
                files.append(p)
    print(f"[INFO] 扫描完成，共发现 {len(files)} 张图片（文件总数 {scanned}）。")
    if limit is not None:
        files = files[:limit]
    return files

# 排除 Screenshot 图片
def is_screenshot(path: Path) -> bool:
    s = str(path)
    return "screenshot" in s.lower()


def filter_unscored(conn: sqlite3.Connection, paths: list[Path]) -> list[Path]:
    if not paths:
        return []

    cur = conn.cursor()
    placeholders = ",".join("?" for _ in paths)
    rows = cur.execute(
        f"SELECT path FROM photo_scores WHERE path IN ({placeholders})",
        [str(p) for p in paths],
    ).fetchall()
    already = {row[0] for row in rows}
    return [p for p in paths if str(p) not in already]


def filter_missing_fields(conn: sqlite3.Connection, paths: list[Path]) -> list[tuple[Path, str]]:
    """筛选数据库中缺少 caption 或 side_caption 的图片
    返回: [(path, missing_field_type)] 列表
    missing_field_type: 'both', 'caption_only', 'side_caption_only'
    """
    if not paths:
        return []

    cur = conn.cursor()
    placeholders = ",".join("?" for _ in paths)
    rows = cur.execute(
        f"""SELECT path, caption, side_caption 
            FROM photo_scores 
            WHERE path IN ({placeholders})
               AND (caption IS NULL OR TRIM(caption) = '' OR side_caption IS NULL OR TRIM(side_caption) = '')""",
        [str(p) for p in paths],
    ).fetchall()
    
    result = []
    for row in rows:
        path, caption, side_caption = row
        caption_empty = not caption or not str(caption).strip()
        side_caption_empty = not side_caption or not str(side_caption).strip()
        
        if caption_empty and side_caption_empty:
            result.append((Path(path), 'both'))
        elif caption_empty:
            result.append((Path(path), 'caption_only'))
        elif side_caption_empty:
            result.append((Path(path), 'side_caption_only'))
    
    return result


def _convert_gps_to_deg(value):
    try:
        d, m, s = value
        return float(d[0]) / float(d[1]) + float(m[0]) / float(m[1]) / 60.0 + float(s[0]) / float(s[1]) / 3600.0
    except Exception:
        return None


def read_gps_with_exiftool(path: Path):
    if not EXIFTOOL_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            ["exiftool", "-n", "-json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        # 没装 exiftool，则直接跳过
        return None
    except subprocess.CalledProcessError:
        return None

    try:
        data = json.loads(result.stdout)[0]
    except Exception:
        return None

    lat = data.get("GPSLatitude")
    lon = data.get("GPSLongitude")
    alt = data.get("GPSAltitude")
    if lat is None or lon is None:
        return None
    return {
        "lat": float(lat),
        "lon": float(lon),
        "alt": float(alt) if alt is not None else None,
    }


def read_exif(path: Path) -> dict:
    info: dict = {}
    try:
        img = Image.open(path)
        try:
            width, height = img.size
            info["width"] = int(width)
            info["height"] = int(height)
            if width > height:
                info["orientation"] = "landscape"
            elif height > width:
                info["orientation"] = "portrait"
            else:
                info["orientation"] = "square"
        except Exception:
            pass
        exif_raw = img._getexif() or {}
    except Exception:
        return info

    exif = {}
    for tag_id, value in exif_raw.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        exif[tag] = value

    # 基本字段
    info["datetime"] = exif.get("DateTimeOriginal") or exif.get("DateTime")
    info["make"] = exif.get("Make")
    info["model"] = exif.get("Model")
    info["iso"] = exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity")
    info["exposure_time"] = exif.get("ExposureTime")
    info["f_number"] = exif.get("FNumber")
    info["focal_length"] = exif.get("FocalLength")

    gps_info = exif.get("GPSInfo")
    lat = lon = None
    if isinstance(gps_info, dict):
        # GPSInfo 的 key 可能是数字，需要映射
        gps_tags = {}
        for k, v in gps_info.items():
            name = ExifTags.GPSTAGS.get(k, k)
            gps_tags[name] = v

        lat_ref = gps_tags.get("GPSLatitudeRef")
        lat_raw = gps_tags.get("GPSLatitude")
        lon_ref = gps_tags.get("GPSLongitudeRef")
        lon_raw = gps_tags.get("GPSLongitude")

        if lat_raw and lat_ref:
            lat = _convert_gps_to_deg(lat_raw)
            if lat is not None and lat_ref in ["S", "s"]:
                lat = -lat
        if lon_raw and lon_ref:
            lon = _convert_gps_to_deg(lon_raw)
            if lon is not None and lon_ref in ["W", "w"]:
                lon = -lon

    info["gps_lat"] = lat
    info["gps_lon"] = lon

    if info.get("gps_lat") is None or info.get("gps_lon") is None:
        gps = read_gps_with_exiftool(path)
        if gps is not None:
            info["gps_lat"] = gps["lat"]
            info["gps_lon"] = gps["lon"]
            if gps.get("alt") is not None:
                info["gps_alt"] = gps["alt"]

    return info


def in_home(lat: float | None, lon: float | None) -> bool:
    """判断是否在“本地/常驻地”范围内。"""
    if lat is None or lon is None:
        return False
    try:
        d = haversine_km(float(lat), float(lon), float(HOME_LAT), float(HOME_LON))
        return d <= float(HOME_RADIUS_KM)
    except Exception:
        return False


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


import csv
import math
from typing import Dict, List, Tuple, Optional

CityRecord = Tuple[float, float, str, str]  # (lat, lon, name_zh, name_en)

_CITY_CACHE_CITIES: List[CityRecord] | None = None
_CITY_CACHE_GRID: Dict[Tuple[int, int], List[int]] | None = None

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c

def grid_key(lat: float, lon: float) -> Tuple[int, int]:
    gx = int(math.floor(lat / CITY_GRID_DEG))
    gy = int(math.floor(lon / CITY_GRID_DEG))
    return gx, gy

def load_world_cities(csv_path: Path) -> Tuple[List[CityRecord], Dict[Tuple[int, int], List[int]]]:
    if not csv_path.exists():
        raise SystemExit(f"[FATAL] 找不到城市索引文件: {csv_path}")

    cities: List[CityRecord] = []
    grid_index: Dict[Tuple[int, int], List[int]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float((row.get("lat") or "").strip())
                lon = float((row.get("lon") or "").strip())
            except Exception:
                continue
            name_en = (row.get("name_en") or "").strip()
            name_zh = (row.get("name_zh") or "").strip()
            cities.append((lat, lon, name_zh, name_en))

    for idx, (lat, lon, name_zh, name_en) in enumerate(cities):
        key = grid_key(lat, lon)
        grid_index.setdefault(key, []).append(idx)

    print(f"[INFO] 已加载中文城市库: {csv_path}")
    return cities, grid_index

def find_nearest_city(
    lat: float,
    lon: float,
    cities: List[CityRecord],
    grid_index: Dict[Tuple[int, int], List[int]],
    max_km: float = 80.0,
) -> str:
    if not cities:
        return ""

    gx, gy = grid_key(lat, lon)

    def collect_candidates(radius: int) -> List[int]:
        cand: List[int] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                bucket = grid_index.get((gx + dx, gy + dy))
                if bucket:
                    cand.extend(bucket)
        return cand

    candidates = collect_candidates(radius=1)
    if not candidates:
        candidates = collect_candidates(radius=2)
    if not candidates:
        return ""

    best_idx: Optional[int] = None
    best_dist = float("inf")

    for idx in candidates:
        city_lat, city_lon, name_zh, name_en = cities[idx]
        d = haversine_km(lat, lon, city_lat, city_lon)
        if d < best_dist:
            best_dist = d
            best_idx = idx

    if best_idx is None or best_dist > max_km:
        return ""

    _, _, name_zh, name_en = cities[best_idx]
    return name_zh or name_en or ""

def get_city_resolver():
    global _CITY_CACHE_CITIES, _CITY_CACHE_GRID
    if _CITY_CACHE_CITIES is None or _CITY_CACHE_GRID is None:
        _CITY_CACHE_CITIES, _CITY_CACHE_GRID = load_world_cities(WORLD_CITIES_CSV)

    def resolve(lat: float | None, lon: float | None) -> str:
        if lat is None or lon is None:
            return ""
        return find_nearest_city(lat, lon, _CITY_CACHE_CITIES, _CITY_CACHE_GRID, max_km=CITY_MAX_DISTANCE_KM)

    return resolve


def call_vlm(image_path: Path) -> dict:
    try:
        img_b64 = encode_image_to_b64(image_path)
    except Exception as e:
        raise RuntimeError(f"读取图片失败：{e}")

    exif_info = read_exif(image_path)
    exif_json = json.dumps(exif_info, ensure_ascii=False, default=str)

    system_prompt = (
        "你是一个“个人相册照片评估助手”，擅长理解真实照片的内容，并从回忆价值和美观角度打分。\n"
        "你会收到一张照片（以 base64 形式提供），你的任务是：\n"
        "1）用中文详细描述照片内容（80~200 字），\n"
        "2）判断照片的大致类型：人物/孩子/猫咪/家庭/旅行/风景/美食/宠物/日常/文档/杂物/其他，一张照片可以有不止一个类型。\n"
        "3）给出 0~100 的“值得回忆度” memory_score（精确到一位小数），\n"
        "4）给出 0~100 的“美观程度” beauty_score（精确到一位小数），\n"
        "5）用简短中文 reason 解释原因（不超过 40 字）。\n\n"

        "【值得回忆度（memory_score）评分方法】\n"
        "请先按照值得回忆的程度，先确定照片的'得分区间'，再进行精调：\n"
        "如何判定值得回忆度（memory_score）的得分区间：\n"
        "- 垃圾/随手拍/无意义记录：40.0 分以下（常见为 0~25；若还能勉强辨认但无故事，也不要超过 39.9）。\n"
        "- 稍微有点可回忆价值：以 65.0 分为中心（大多落在 58.1~70.3）。\n"
        "- 不错的回忆价值：以 75 分为中心（大多落在 68.7~82.4）。\n"
        "- 特别精彩、强烈值得珍藏：以 85 分为中心（大多落在 79.1~95.9；\n"
        "如何继续精调memory_score得分（若同时符合几条加分项，加分可叠加）：\n"
        "- 人物与关系：画面中含有面积较大的人脸，有人物互动，或属于合影 → 大幅提高评分；\n"
        "- 事件性：生日/聚会/仪式/舞台/明显事件 → 少许提高评分；\n"
        "- 稀缺性与不可复现：明显“这一刻很难再来一次” → 大幅提高评分；\n"
        "- 情绪强度：笑、哭、惊喜、拥抱、互动、氛围强 → 少许提高评分；\n"
        "- 信息密度：画面能讲清楚发生了什么 → 微微提高评分；\n"
        "- 优美风景：画面中含有壮丽的自然风光，或精美、有秩序感的构图 → 少许提高评分；\n"
        "- 旅行意义：异地、地标、旅途情景 → 少许提高评分。\n\n"
        "- 画质：画面不清晰、模糊、有残影、虚焦 → 微微降低评分。\n\n"

        "【重点照片的处理】\n"
        "如果画面中含有：孩子/猫咪/宠物题材，这些主题更容易产生高回忆价值，请直接以75分为中心，并大幅提高评分”。\n"

        "【明显低价值图片的处理】\n"
        "对以下低价值图片，必须将 memory_score 压低到 0~25（最多不超过 39）。\n"
        "- 裸露、低俗、色情或违反公序良俗的图片。\n\n"
        "- 账单、收据、广告、随手拍的杂物、测试图片、屏幕截图等。\n\n"
        
        "【美观分（beauty_score）评分方法】\n"
        "美观分只评价视觉：构图、光线、清晰度、色彩、主体突出。\n"
        "不要被“孩子/猫/旅行”主题绑架美观分：主题不等于好看。\n"

        "请严格只输出 JSON，格式如下：\n"
        "{\n"
        "  \"caption\": \"……\",\n"
        "  \"type\": \"人物/家庭/旅行/…… 可以带多个type\",\n"
        "  \"memory_score\": 0.0-100.0 的数字, 精确到 1 位小数\n"
        "  \"beauty_score\": 0.0-100.0 的数字, 精确到 1 位小数\n"
        "  \"reason\": \"不超过 60 字的中文理由\"\n"
        "}\n"
        "不要输出任何多余文字，不要markdown语法，不要以```json开头，一定要“{”开头，一定要“}”结尾。"
    )

    user_text = (
        "下面是照片的内容，请结合图像本身完成上述任务。\n"
    )

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
        "repeat_penalty": 1,
        "stop":["```"],
        "stream": False,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
    if not resp.ok:
        print("HTTP:", resp.status_code)
        print(resp.text)
        raise RuntimeError(f"LM Studio 请求失败: HTTP {resp.status_code}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except Exception:
        print("[DEBUG] 返回内容：", data)
        raise RuntimeError("解析失败：无法从 choices[0].message.content 读取内容")

    # content 应该是 JSON 字符串
    try:
        obj = json.loads(content)
    except Exception:
        print("[DEBUG] 非 JSON 输出：", content)
        raise RuntimeError("解析失败：模型未按 JSON 输出")

    return obj, exif_info


def main():
    filelist_path = ROOT_DIR / "filelist.txt"

    print("[INFO] 正在扫描图片目录……")
    imgs = list_images()
    filelist_path.write_text("\n".join(str(p) for p in imgs), encoding="utf-8")
    print(f"[INFO] 已更新文件列表 filelist.txt，共 {len(imgs)} 个文件。")
    if not imgs:
        raise SystemExit(f"目录下没有图片文件: {', '.join(str(d) for d in IMAGE_DIRS)}")

    imgs = [p for p in imgs if not is_screenshot(p)]
    if not imgs:
        raise SystemExit("[INFO] 所有图片都被 Screenshot 过滤规则排除了，没有可处理的图片。")

    conn = sqlite3.connect(DB_PATH)
    ensure_table(conn)
    city_resolver = get_city_resolver()

    # =======================
    # 同步删除：NAS/磁盘上已不存在的文件，也从数据库里删除
    # 只处理当前 IMAGE_DIRS 前缀下的记录，避免误删其它历史路径。
    # =======================

    try:
        # 用临时表避免 IN (...) 过长导致的 SQLite 参数上限问题
        conn.execute("DROP TABLE IF EXISTS _temp_existing_paths")
        conn.execute("CREATE TEMP TABLE _temp_existing_paths (path TEXT PRIMARY KEY)")

        # 批量插入当前扫描到的文件列表
        CHUNK = 2000
        total_files = len(imgs)
        inserted = 0
        for i in range(0, total_files, CHUNK):
            chunk = imgs[i : i + CHUNK]
            conn.executemany(
                "INSERT OR IGNORE INTO _temp_existing_paths(path) VALUES (?)",
                [(str(p),) for p in chunk],
            )
            inserted += len(chunk)
            if inserted % 10000 == 0:
                print(f"[CLEAN] 已写入存在文件清单：{inserted}/{total_files} …")

        # 删除：数据库里有记录，但磁盘上已不存在的文件
        cur_clean = conn.cursor()
        
        # 构建 WHERE 条件，检查所有的目录前缀
        where_conditions = []
        params = []
        for img_dir in IMAGE_DIRS:
            where_conditions.append("path LIKE ?")
            params.append(str(img_dir) + "%")
        
        # 构建完整的 WHERE 子句
        if where_conditions:
            where_clause = " OR ".join(where_conditions)
            
            # 计算删除前的记录数
            count_query = f"SELECT COUNT(*) FROM photo_scores WHERE {where_clause}"
            before_cnt = cur_clean.execute(count_query, params).fetchone()[0]

            # 构建删除语句
            delete_query = f"""
            DELETE FROM photo_scores
            WHERE ({where_clause})
              AND NOT EXISTS (
                    SELECT 1 FROM _temp_existing_paths t
                    WHERE t.path = photo_scores.path
              )
            """
            
            cur_clean.execute(delete_query, params)
        else:
            before_cnt = 0
        deleted = cur_clean.rowcount if cur_clean.rowcount is not None else 0
        conn.commit()

        after_cnt = cur_clean.execute(
            "SELECT COUNT(*) FROM photo_scores WHERE path LIKE ?",
            (image_dir_prefix + "%",),
        ).fetchone()[0]

        if deleted > 0:
            print(f"[CLEAN] 已同步删除 {deleted} 条数据库残留记录（当前目录：{before_cnt} → {after_cnt}）。")
        else:
            print("[CLEAN] 数据库与磁盘文件一致，无需清理。")

    except Exception as e:
        # 清理失败不应影响主流程
        print(f"[WARN] 同步清理数据库残留记录失败（已忽略，不影响主流程）：{e}")

    cur_test = conn.cursor()
    # 只统计当前 IMAGE_DIRS 下的已分析照片，避免数据库里其它路径/历史残留影响进度计算
    if IMAGE_DIRS:
        # 构建 WHERE 条件，检查所有的目录前缀
        where_conditions = []
        params = []
        for img_dir in IMAGE_DIRS:
            where_conditions.append("path LIKE ?")
            params.append(str(img_dir) + "%")
        
        where_clause = " OR ".join(where_conditions)
        counted = cur_test.execute(
            f"SELECT COUNT(*) FROM photo_scores WHERE {where_clause}",
            params,
        ).fetchone()[0]
    else:
        counted = 0
    print(f"[INFO] 数据库中已有 {counted} 张已分析照片（仅统计当前目录）。")

    # 获取完全未分析的图片
    unscored_paths = filter_unscored(conn, imgs)
        
    # 获取缺少字段的图片
    missing_field_items = filter_missing_fields(conn, imgs)
        
    if not unscored_paths and not missing_field_items:
        print("[INFO] 所有图片都已经完整分析过了。")
        conn.close()
        return
    
    # 合并处理列表：未分析的图片优先
    target_items = []
        
    # 添加未分析的图片（标记为 'new'）
    for path in unscored_paths:
        target_items.append((path, 'new'))
        
    # 添加缺少字段的图片
    target_items.extend(missing_field_items)
        
    if BATCH_LIMIT is not None:
        target_items = target_items[:BATCH_LIMIT]
    
    # 进度条口径：以"本次启动时的快照"为准。
    # total = 已分析(当前目录) + 本次待处理
    already_done = counted
    total = already_done + len(target_items)
    print(f"[INFO] 本次准备处理 {len(target_items)} 张图片（快照总数 {total}，已分析 {already_done}）。")
    print(f"     - 其中 {len(unscored_paths)} 张完全未分析")
    print(f"     - 其中 {len(missing_field_items)} 张缺少部分字段")

    cur = conn.cursor()
    start_time = time.time()

    for idx, (path, item_type) in enumerate(target_items, start=1):
        t_photo_start = time.perf_counter()
        sep = "=" * 60
        print("\n" + sep)
        if item_type == 'new':
            print(f"[{idx}/{len(target_items)}] 处理(新图片): {path}")
        else:
            missing_desc = {
                'both': '缺少caption和side_caption',
                'caption_only': '缺少caption',
                'side_caption_only': '缺少side_caption'
            }
            print(f"[{idx}/{len(target_items)}] 处理({missing_desc[item_type]}): {path}")
        
        # 根据不同类型采用不同的处理逻辑
        if item_type == 'new':
            # 完全新图片：执行完整分析流程
            try:
                result, exif_info = call_vlm(path)
            except Exception as e:
                print(f"[WARN] 调用模型失败: {e}")
                continue
            t_after_vlm = time.perf_counter()
            vlm_cost = t_after_vlm - t_photo_start

            caption = str(result.get("caption", "")).strip()
            ptype = str(result.get("type", "")).strip()
            try:
                memory_score = float(result.get("memory_score", 0.0))
            except Exception:
                memory_score = 0.0
            try:
                beauty_score = float(result.get("beauty_score", 0.0))
            except Exception:
                beauty_score = 0.0
            reason = str(result.get("reason", "")).strip()

            side_caption = generate_side_caption(path)
            t_after_side = time.perf_counter()
            side_cost = t_after_side - t_after_vlm
        else:
            # 缺少部分字段的图片：只补充缺失的部分
            # 先从数据库获取现有信息
            existing_row = cur.execute(
                "SELECT caption, type, memory_score, beauty_score, reason, side_caption, exif_json FROM photo_scores WHERE path = ?",
                (str(path),)
            ).fetchone()
            
            if not existing_row:
                print(f"[WARN] 数据库中找不到记录: {path}")
                continue
                
            existing_caption, existing_type, existing_memory, existing_beauty, existing_reason, existing_side, existing_exif = existing_row
            
            # 初始化变量
            caption = str(existing_caption or "").strip()
            ptype = str(existing_type or "").strip()
            try:
                memory_score = float(existing_memory) if existing_memory is not None else 0.0
            except Exception:
                memory_score = 0.0
            try:
                beauty_score = float(existing_beauty) if existing_beauty is not None else 0.0
            except Exception:
                beauty_score = 0.0
            reason = str(existing_reason or "").strip()
            side_caption = str(existing_side or "").strip()
            
            # 读取EXIF信息（用于补充side_caption时可能需要）
            try:
                exif_info = read_exif(path)
            except Exception:
                exif_info = {}
            
            # 根据缺失类型补充相应字段
            if item_type in ['both', 'caption_only']:
                # 需要补充caption相关字段
                try:
                    result, exif_info = call_vlm(path)
                    caption = str(result.get("caption", "")).strip()
                    ptype = str(result.get("type", "")).strip()
                    try:
                        memory_score = float(result.get("memory_score", 0.0))
                    except Exception:
                        memory_score = 0.0
                    try:
                        beauty_score = float(result.get("beauty_score", 0.0))
                    except Exception:
                        beauty_score = 0.0
                    reason = str(result.get("reason", "")).strip()
                    t_after_vlm = time.perf_counter()
                except Exception as e:
                    print(f"[WARN] 调用模型失败: {e}")
                    # 如果调用失败，保留原有值
                    t_after_vlm = time.perf_counter()
                    pass
            else:
                # 不需要调用VLM，设置时间为当前时间
                t_after_vlm = time.perf_counter()
            
            if item_type in ['both', 'side_caption_only']:
                # 需要补充side_caption
                new_side_caption = generate_side_caption(path)
                if new_side_caption:
                    side_caption = new_side_caption
                t_after_side = time.perf_counter()
            else:
                # 不需要生成side_caption，设置时间为VLM之后的时间
                t_after_side = t_after_vlm

        width = exif_info.get("width")
        height = exif_info.get("height")
        orientation = exif_info.get("orientation")

        exif_datetime = exif_info.get("datetime")
        exif_make = exif_info.get("make")
        exif_model = exif_info.get("model")

        def _to_int(v):
            try:
                if v is None:
                    return None
                return int(v)
            except Exception:
                return None

        def _to_float(v):
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        exif_iso = _to_int(exif_info.get("iso"))
        exif_exposure_time = _to_float(exif_info.get("exposure_time"))
        exif_f_number = _to_float(exif_info.get("f_number"))
        exif_focal_length = _to_float(exif_info.get("focal_length"))
        exif_gps_lat = _to_float(exif_info.get("gps_lat"))
        exif_gps_lon = _to_float(exif_info.get("gps_lon"))
        exif_gps_alt = _to_float(exif_info.get("gps_alt"))

        if exif_gps_lat is not None and exif_gps_lon is not None:
            exif_city = city_resolver(exif_gps_lat, exif_gps_lon)
        else:
            exif_city = ""

        # 如果有 GPS 信息且不在本地范围内，略微提高回忆分（最多 +5，且不超过 100 分）
        lat = exif_info.get("gps_lat")
        lon = exif_info.get("gps_lon")
        if lat is not None and lon is not None and not in_home(lat, lon):
            memory_score = min(memory_score + 5.0, 100.0)

        exif_json = json.dumps(exif_info, ensure_ascii=False, default=str)
        
        # 准备raw_json（只有在调用了VLM时才有result变量）
        raw_json = ""
        if 'result' in locals():
            raw_json = json.dumps(result, ensure_ascii=False)
        else:
            # 从数据库获取现有的raw_json
            existing_raw_json = cur.execute(
                "SELECT raw_json FROM photo_scores WHERE path = ?",
                (str(path),)
            ).fetchone()
            raw_json = existing_raw_json[0] if existing_raw_json and existing_raw_json[0] else ""

        print(f"  类型    ：{ptype}")
        print(f"  回忆分  ：{memory_score:.1f}")
        print(f"  美观分  ：{beauty_score:.1f}")
        if side_caption:
            print(f"  一句话文案：{side_caption}")
        else:
            print("  一句话文案：(无)")
        print(f"  画面描述：{caption}")
        print(f"  理由    ：{reason}")

        cur.execute(
            """
            INSERT OR REPLACE INTO photo_scores
            (path, caption, type, memory_score, beauty_score, reason,
             width, height, orientation, used_at,
             exif_json, raw_json,
             exif_datetime, exif_make, exif_model,
             exif_iso, exif_exposure_time, exif_f_number, exif_focal_length,
             exif_gps_lat, exif_gps_lon, exif_gps_alt, side_caption, exif_city)
            VALUES (?, ?, ?, ?, ?, ?,
                    ?, ?, ?, COALESCE((SELECT used_at FROM photo_scores WHERE path = ?), NULL),
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
            """,
            (
                str(path),
                caption,
                ptype,
                memory_score,
                beauty_score,
                reason,
                width,
                height,
                orientation,
                str(path),
                exif_json,
                raw_json,
                exif_datetime,
                exif_make,
                exif_model,
                exif_iso,
                exif_exposure_time,
                exif_f_number,
                exif_focal_length,
                exif_gps_lat,
                exif_gps_lon,
                exif_gps_alt,
                side_caption,
                exif_city,
            ),
        )
        conn.commit()
        t_photo_end = time.perf_counter()
        total_cost = t_photo_end - t_photo_start
        # pretty timing summary

        # 进度条与预估时间（以本次启动时的快照为准，不受运行中新增照片影响）
        processed_now = already_done + idx

        denom = total if total > 0 else 1
        progress = processed_now / denom
        # 夹紧，确保不会超过 100%
        if progress < 0:
            progress = 0.0
        if progress > 1:
            progress = 1.0

        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - start_time
        avg_per = elapsed / idx if idx > 0 else 0
        remaining = max(total - processed_now, 0)
        eta = format_eta(remaining * avg_per) if avg_per > 0 else "00:00:00"

        print(f"[进度] {bar} {progress*100:5.1f}%  {processed_now}/{total}  本张耗时 {total_cost:4.1f}s  预计剩余 {eta} ")

    conn.close()
    print("\n[完成] 本批次处理完成。")


if __name__ == "__main__":
    require_exiftool()
    main()