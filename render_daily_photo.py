#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
每日相册渲染脚本：
- 从 photos.db / photo_scores 中选出一张"历史上的今天"照片
- 按 InkTime 模拟器的布局渲染到 1080x1920
- 用 LXGWHeartSerifMN.ttf 把文案 / 日期 / 地点都画到图上
- 专为彩色液晶显示屏设计，无抖动算法
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import json
import datetime as dt
import os
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import config as cfg


TODAY = dt.date.today()

# === 路径配置（来自 config.py） ===
ROOT_DIR = Path(__file__).resolve().parent

DB_PATH = Path(str(getattr(cfg, "DB_PATH", "photos.db") or "photos.db")).expanduser()
if not DB_PATH.is_absolute():
    DB_PATH = (ROOT_DIR / DB_PATH).resolve()

BIN_OUTPUT_DIR = Path(str(getattr(cfg, "BIN_OUTPUT_DIR", "output/inktime") or "output/inktime")).expanduser()
if not BIN_OUTPUT_DIR.is_absolute():
    BIN_OUTPUT_DIR = (ROOT_DIR / BIN_OUTPUT_DIR).resolve()
BIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path(str(getattr(cfg, "FONT_PATH", "") or "")).expanduser()
if str(FONT_PATH) and not FONT_PATH.is_absolute():
    FONT_PATH = (ROOT_DIR / FONT_PATH).resolve()

MEMORY_THRESHOLD = float(getattr(cfg, "MEMORY_THRESHOLD", 70.0) or 70.0)
DAILY_PHOTO_QUANTITY = int(getattr(cfg, "DAILY_PHOTO_QUANTITY", 5) or 5)

# 画布尺寸
CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

# 底部文字区域高度
TEXT_AREA_HEIGHT = 180


# ========== DB 与 EXIF 处理 ==========

def extract_date_from_exif(exif_json: Optional[str]) -> str:
    """
    从 EXIF JSON 中提取拍摄日期，返回 YYYY-MM-DD 格式，失败则返回空字符串。
    逻辑与 review_web.py 中保持一致。
    """
    if not exif_json:
        return ""
    try:
        data = json.loads(exif_json)
    except Exception:
        return ""
    dt_str = data.get("datetime")
    if not dt_str:
        return ""
    try:
        date_part = str(dt_str).split()[0]
        parts = date_part.replace(":", "-").split("-")
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
    except Exception:
        return ""
    return ""


def load_sim_rows() -> List[Dict[str, Any]]:
    """
    加载 InkTime 用的核心字段：
    - path: 照片路径
    - exif_json: 用于解析日期 / GPS
    - side_caption: 文案
    - memory_score: 回忆度
    - exif_gps_lat / exif_gps_lon / exif_city: 地点信息（纯本地，不上网）
    """
    if not DB_PATH.exists():
        raise SystemExit(f"找不到数据库文件: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    rows = c.execute(
        """
        SELECT path,
               exif_json,
               side_caption,
               memory_score,
               exif_gps_lat,
               exif_gps_lon,
               exif_city
        FROM photo_scores
        WHERE exif_json IS NOT NULL
        """
    ).fetchall()
    conn.close()

    items: List[Dict[str, Any]] = []
    for path, exif_json, side_caption, memory_score, gps_lat, gps_lon, exif_city in rows:
        date_str = extract_date_from_exif(exif_json)
        if not date_str:
            continue
        # 再次兜底过滤 Screenshot 等
        if "screenshot" in str(path).lower():
            continue

        try:
            y, m, d = map(int, date_str.split("-"))
        except Exception:
            continue
        md = f"{m:02d}-{d:02d}"

        item = {
            "path": str(path),
            "date": date_str,  # YYYY-MM-DD
            "md": md,          # MM-DD
            "side": side_caption or "",
            "memory": float(memory_score) if memory_score is not None else -1.0,
            "lat": gps_lat,
            "lon": gps_lon,
            "city": exif_city or "",
        }
        items.append(item)

    return items


# ========== "历史上的今天"选片 ==========

def md_to_day_of_year(md: str) -> Optional[int]:
    """把 'MM-DD' 转成非闰年的第几天（1~365）。"""
    try:
        m, d = map(int, md.split("-"))
        days_before = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        if m < 1 or m > 12:
            return None
        return days_before[m] + d
    except Exception:
        return None


def day_of_year_to_md(day: int) -> str:
    # 选一个非闰年（2001/2005 随便），只依赖 day-of-year。
    base = dt.date(2001, 1, 1) + dt.timedelta(days=day - 1)
    return f"{base.month:02d}-{base.day:02d}"


def choose_photo_for_today(items: List[Dict[str, Any]], today: dt.date) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    选片规则（按月日）：
    - 以 today 的月日为目标，例如 12 月 2 日 -> "12-02"
    - 在所有年份该月日的照片中，找 memory > MEMORY_THRESHOLD 的候选，随机选一张
    - 如果该月日没有任何 > 阈值的，则往前一天（月日）继续找（12-01, 11-30, ...），最多回溯 365 天
    - 如果整个 365 天都没有任何 > 阈值的照片，则在全局中选 memory 最大的一张作为兜底
    """

    if not items:
        raise RuntimeError("没有任何可用照片")

    # 按 md 分组
    by_md: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        md = it["md"]
        by_md.setdefault(md, []).append(it)

    # 每组内按 memory 从高到低排序
    for arr in by_md.values():
        arr.sort(key=lambda x: x.get("memory", -1.0), reverse=True)

    target_md = f"{today.month:02d}-{today.day:02d}"
    target_doy = md_to_day_of_year(target_md)
    if target_doy is None:
        raise RuntimeError(f"无法解析今天的月日: {target_md}")

    import random

    for offset in range(0, 365):
        doy = target_doy - offset
        if doy <= 0:
            doy += 365
        md = day_of_year_to_md(doy)

        arr = by_md.get(md, [])
        if not arr:
            continue
        candidates = [p for p in arr if p.get("memory", -1.0) > MEMORY_THRESHOLD]
        if not candidates:
            continue

        chosen = random.choice(candidates)
        info = {
            "target_md": target_md,
            "used_md": md,
            "day_offset": -offset,
            "candidate_count": len(candidates),
            "total_count_md": len(arr),
            "threshold": MEMORY_THRESHOLD,
            "fallback_global_max": False,
        }
        return chosen, info

    global_best = max(items, key=lambda x: x.get("memory", -1.0))
    info = {
        "target_md": target_md,
        "used_md": global_best["md"],
        "day_offset": None,
        "candidate_count": 1,
        "total_count_md": len(by_md.get(global_best["md"], [])),
        "threshold": MEMORY_THRESHOLD,
        "fallback_global_max": True,
    }
    return global_best, info


def choose_photos_for_today(items: List[Dict[str, Any]], today: dt.date, count: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    选片规则（多张版，按月日）：
    - 以 today 的月日为目标，例如 12 月 2 日 -> "12-02"
    - 在所有年份该月日的照片中，找 memory > MEMORY_THRESHOLD 的候选，尽量随机选 count 张
    - 如果该月日没有任何 > 阈值的，则往前一天（月日）继续找（12-01, 11-30, ...），最多回溯 365 天
    - 如果整个 365 天都没有任何 > 阈值的照片，则在全局中选回忆度最高的若干张作为兜底
    """
    if not items:
        raise RuntimeError("没有任何可用照片")

    # 按 md 分组
    by_md: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        md = it["md"]
        by_md.setdefault(md, []).append(it)

    # 每组内按 memory 从高到低排序
    for arr in by_md.values():
        arr.sort(key=lambda x: x.get("memory", -1.0), reverse=True)

    target_md = f"{today.month:02d}-{today.day:02d}"
    target_doy = md_to_day_of_year(target_md)
    if target_doy is None:
        raise RuntimeError(f"无法解析今天的月日: {target_md}")

    import random

    for offset in range(0, 365):
        doy = target_doy - offset
        if doy <= 0:
            doy += 365
        md = day_of_year_to_md(doy)

        arr = by_md.get(md, [])
        if not arr:
            continue
        candidates = [p for p in arr if p.get("memory", -1.0) > MEMORY_THRESHOLD]
        if not candidates:
            continue

        # 随机选不重复的多张
        if len(candidates) >= count:
            chosen_list = random.sample(candidates, count)
        else:
            # 候选不足 count 张，用该日剩余的高分照片补齐
            chosen_list = list(candidates)
            for extra in arr:
                if extra in chosen_list:
                    continue
                chosen_list.append(extra)
                if len(chosen_list) >= count:
                    break

        info = {
            "target_md": target_md,
            "used_md": md,
            "day_offset": -offset,
            "candidate_count": len(candidates),
            "total_count_md": len(arr),
            "threshold": MEMORY_THRESHOLD,
            "fallback_global_max": False,
        }
        return chosen_list, info

    # 兜底：全局回忆度最高的若干张
    sorted_all = sorted(items, key=lambda x: x.get("memory", -1.0), reverse=True)
    chosen_list = sorted_all[:count]
    info = {
        "target_md": target_md,
        "used_md": chosen_list[0]["md"] if chosen_list else "",
        "day_offset": None,
        "candidate_count": len(chosen_list),
        "total_count_md": len(items),
        "threshold": MEMORY_THRESHOLD,
        "fallback_global_max": True,
    }
    return chosen_list, info


# ========== 绘制 + 后处理 ==========

def wrap_text_chinese(draw: ImageDraw.ImageDraw,
                      text: str,
                      font: ImageFont.FreeTypeFont,
                      max_width: int,
                      max_lines: int) -> List[str]:
    """
    简单中文按字符宽度折行。
    """
    if not text:
        return []
    lines: List[str] = []
    line = ""
    for ch in text:
        test = line + ch
        w = draw.textlength(test, font=font)
        if w <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = ch
            if len(lines) >= max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line)
    return lines


def format_date_display(date_str: str) -> str:
    """
    "YYYY-MM-DD" -> "YYYY.M.D"
    """
    if not date_str:
        return ""
    parts = date_str.split("-")
    if len(parts) < 3:
        return date_str
    y = parts[0]
    try:
        m = str(int(parts[1]))
        d = str(int(parts[2]))
    except Exception:
        return date_str
    return f"{y}.{m}.{d}"


def format_location(lat, lon, city: str) -> str:
    """
    地点字符串：
    - 有 city 用 city
    - 否则如果有 lat/lon，用 "lat, lon"（5 位小数）
    - 否则空字符串（不写"未知地点"）
    """
    if city and str(city).strip():
        return str(city).strip()
    if lat is None or lon is None:
        return ""
    try:
        return f"{float(lat):.5f}, {float(lon):.5f}"
    except Exception:
        return ""


def render_image(item: Dict[str, Any]) -> Image.Image:
    """
    根据选中的 item 渲染一张 1080x1920 的 RGB 图像（竖屏）：
    - Contain 模式：保持图片比例完整显示在画布内
    - 使用渐变背景填充空白区域
    - 底部文字区域覆盖在照片上层，带半透明白色背景
    """
    # ---------- 加载原图并按 EXIF 方向纠正 ----------
    img_path = Path(item["path"])
    if not img_path.exists():
        raise RuntimeError(f"图片不存在：{img_path}")
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    img_w, img_h = img.size
    if img_w == 0 or img_h == 0:
        raise RuntimeError(f"图片尺寸非法：{img.size}")

    # ---------- 照片区域（Contain 模式） ----------
    padding_x = 48

    # Contain 模式：使用 min() 保证图片完整显示在画布内
    scale = min(CANVAS_WIDTH / img_w, CANVAS_HEIGHT / img_h)
    draw_w = int(img_w * scale)
    draw_h = int(img_h * scale)

    img_resized = img.resize((draw_w, draw_h), Image.LANCZOS)

    # 计算居中位置
    left = (CANVAS_WIDTH - draw_w) // 2
    top = (CANVAS_HEIGHT - draw_h) // 2

    # ---------- 创建电影海报质感的模糊背景 ----------
    # 步骤1: 创建模糊背景层（用原图的模糊版本）
    bg_img = img.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.LANCZOS)
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=20))
    
    # 步骤2: 压暗背景，让前景更突出
    darken_factor = 0.8
    bg_img = ImageEnhance.Brightness(bg_img).enhance(darken_factor)
    
    # 创建画布并粘贴模糊背景
    canvas = bg_img.copy()
    
    # 步骤3: 在模糊背景上粘贴清晰的前景图片（居中）
    canvas.paste(img_resized, (left, top))
    
    draw = ImageDraw.Draw(canvas)

    # ---------- 底部文字区域（覆盖在照片上层） ----------
    text_area_top = CANVAS_HEIGHT - TEXT_AREA_HEIGHT
    text_width = CANVAS_WIDTH - 2 * padding_x

    # 创建半透明白色背景
    text_bg = Image.new("RGBA", (CANVAS_WIDTH, TEXT_AREA_HEIGHT), (255, 255, 255, 204))
    canvas.paste(text_bg, (0, text_area_top), text_bg)

    try:
        font_big = ImageFont.truetype(str(FONT_PATH), 44)
        font_small = ImageFont.truetype(str(FONT_PATH), 36)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    side_text = item.get("side") or ""

    # 文案：最多两行，从 text_area_top 开始
    y = text_area_top + 20
    if side_text:
        lines = wrap_text_chinese(draw, side_text, font_big, text_width, max_lines=2)
        for line in lines:
            draw.text((padding_x, y), line, font=font_big, fill=(0, 0, 0))
            y += 48

    # 日期 + 地点：固定在底部区域内的第二行
    date_display = format_date_display(item["date"])
    loc_display = format_location(item.get("lat"), item.get("lon"), item.get("city") or "")

    second_line_y = text_area_top + 108
    draw.text((padding_x, second_line_y), date_display, font=font_small, fill=(0, 0, 0))

    loc_w = draw.textlength(loc_display, font=font_small)
    loc_x = padding_x + text_width - loc_w
    if loc_x < padding_x:
        loc_x = padding_x
    draw.text((loc_x, second_line_y), loc_display, font=font_small, fill=(0, 0, 0))

    return canvas


# ========== 主流程 ==========

def main():
    items = load_sim_rows()
    if not items:
        raise SystemExit("没有可用照片（exif_json 为空或解析失败）。")

    photos, info = choose_photos_for_today(items, TODAY, count=DAILY_PHOTO_QUANTITY)

    print("[INFO] 目标月日:", info["target_md"])
    print("[INFO] 实际使用月日:", info["used_md"])
    print("[INFO] 回溯天数(day_offset):", info["day_offset"])
    print("[INFO] 候选数(>阈值):", info["candidate_count"])
    print("[INFO] 当日总数:", info["total_count_md"])
    print("[INFO] 使用兜底全局最大:", info["fallback_global_max"])

    if not photos:
        raise SystemExit("选片结果为空。")

    import shutil

    # 对今天选出的多张照片逐一渲染
    for idx, chosen in enumerate(photos):
        print(f"[INFO] 第 {idx} 张选中照片:", chosen["path"])
        print("[INFO] 拍摄日期:", chosen["date"])
        print("[INFO] 回忆度:", chosen["memory"])
        # 额外调试信息：城市 / 经纬度 / 文案
        print("[DEBUG] 城市:", chosen.get("city", ""))
        print("[DEBUG] 经纬度:", chosen.get("lat"), chosen.get("lon"))
        print("[DEBUG] 文案:", chosen.get("side", ""))

        # 渲染成完整成品图（照片 + 文案 + 日期 + 地点）
        img = render_image(chosen)

        # 保存全彩色 JPG 图像（原始渲染结果），使用最高质量设置
        jpg_path = BIN_OUTPUT_DIR / f"color_{idx}.jpg"
        img.save(jpg_path, "JPEG", quality=100, optimize=True, subsampling=0)
        print(f"[INFO] 已保存全彩色 JPG: {jpg_path}")

        # 保存彩色 PNG 预览
        preview_path = BIN_OUTPUT_DIR / f"preview_{idx}.png"
        img.save(preview_path, "PNG", optimize=True)
        print(f"[OK] 已保存彩色预览 PNG: {preview_path}")

    # 为兼容旧流程，再额外生成 latest.* 指向第 0 张
    first_color = BIN_OUTPUT_DIR / "color_0.jpg"
    first_preview = BIN_OUTPUT_DIR / "preview_0.png"
    latest_color = BIN_OUTPUT_DIR / "latest.jpg"
    latest_preview = BIN_OUTPUT_DIR / "preview.png"

    if first_color.exists():
        shutil.copyfile(first_color, latest_color)
        print(f"[OK] 已更新 latest.jpg -> {first_color.name}")
    if first_preview.exists():
        shutil.copyfile(first_preview, latest_preview)
        print(f"[OK] 已更新 preview.png -> {first_preview.name}")


if __name__ == "__main__":
    main()
