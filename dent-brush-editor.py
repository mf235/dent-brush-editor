#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dent Brush Editor v0.3.1
ブラシでなぞった部分に凹み・食い込み風の陰影と変位を付ける画像編集ツール。

Required libraries:
    PySide6, numpy, opencv-python, Pillow
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

APP_NAME = "Dent Brush Editor"
APP_VERSION = "0.3.1"
SETTINGS_NAME = "dent-brush-editor-settings.json"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_CENTER_LINE_COLOR = "#000000"
PROJECT_FILE_FILTER = "Dent Project (*.dent.json);;JSON (*.json)"

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - runtime guard
    print("numpy is required.", exc)
    raise

try:
    import cv2
except Exception as exc:  # pragma: no cover - runtime guard
    print("opencv-python is required.", exc)
    raise

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - runtime guard
    print("Pillow is required.", exc)
    raise

try:
    from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
    from PySide6.QtGui import (
        QAction,
        QColor,
        QCursor,
        QDragEnterEvent,
        QDropEvent,
        QImage,
        QKeySequence,
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QWheelEvent,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QColorDialog,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSpinBox,
        QSplitter,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - runtime guard
    print("PySide6 is required to run this GUI tool.")
    print("Install example: pip install PySide6 numpy opencv-python Pillow")
    raise


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def settings_path() -> Path:
    return app_dir() / SETTINGS_NAME


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def cv2_read_image_unicode(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像として読み込めません: {path}")
    if img.ndim == 2:
        rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return rgba


def cv2_write_image_unicode(path: Path, rgba: np.ndarray, quality: int = 95) -> None:
    suffix = path.suffix.lower()
    params: List[int] = []
    if suffix in {".jpg", ".jpeg"}:
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), clamp_int(quality, 1, 100)]
    elif suffix == ".webp":
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), clamp_int(quality, 1, 100)]
    else:
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        ext = ".png"
    ok, buf = cv2.imencode(ext, bgra if suffix not in {".jpg", ".jpeg"} else bgr, params)
    if not ok:
        raise ValueError(f"画像エンコードに失敗しました: {path}")
    path.write_bytes(buf.tobytes())


def ndarray_rgba_to_qimage(rgba: np.ndarray) -> QImage:
    if not rgba.flags["C_CONTIGUOUS"]:
        rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    fmt = getattr(QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QImage.Format.Format_RGBA8888
    qimg = QImage(rgba.data, w, h, rgba.strides[0], fmt)
    return qimg.copy()


def qimage_to_rgba(qimg: QImage) -> np.ndarray:
    fmt = getattr(QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QImage.Format.Format_RGBA8888
    img = qimg.convertToFormat(fmt)
    w = img.width()
    h = img.height()
    ptr = img.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, img.bytesPerLine()))[:, : w * 4]
    return arr.reshape((h, w, 4)).copy()


def unique_output_path(src: Optional[Path], suffix: str = ".png") -> Path:
    base_dir = src.parent if src else app_dir()
    stem = src.stem if src else "clipboard_image"
    candidate = base_dir / f"{stem}_dent{suffix}"
    if not candidate.exists():
        return candidate
    for i in range(2, 10000):
        candidate = base_dir / f"{stem}_dent_v{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("保存ファイル名の連番作成に失敗しました。")


@dataclass
class ToolParams:
    brush_size: int = 5
    brush_hardness: int = 62
    brush_opacity: int = 72
    brush_spacing: int = 15
    smoothing: int = 24
    depth: int = 40
    displacement: int = 34
    pull: int = 41
    rim: int = 38
    edge_blur: int = 8
    texture_noise: int = 14
    shadow_strength: int = 10
    highlight_strength: int = 0
    light_angle: int = 14
    shadow_blur: int = 6
    shadow_spread: int = 39
    highlight_width: int = 10
    inner_dark: int = 50
    final_blur_enabled: int = 1
    final_blur_size: int = 0
    final_blur_strength: int = 0
    center_line_enabled: int = 1
    center_line_width: int = 0
    center_line_opacity: int = 100
    jpeg_quality: int = 95
    webp_quality: int = 95

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ToolParams":
        base = cls()
        for key, value in data.items():
            if hasattr(base, key):
                try:
                    setattr(base, key, int(value))
                except Exception:
                    pass
        base.clamp_all()
        return base

    def clamp_all(self) -> None:
        ranges = PARAM_RANGES
        for name, (lo, hi) in ranges.items():
            setattr(self, name, clamp_int(getattr(self, name), lo, hi))




@dataclass
class StrokeRecord:
    erase: bool
    points: List[Tuple[float, float]]
    brush_size: int
    brush_hardness: int
    brush_opacity: int
    brush_spacing: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "erase": bool(self.erase),
            "points": [[float(x), float(y)] for x, y in self.points],
            "brush_size": int(self.brush_size),
            "brush_hardness": int(self.brush_hardness),
            "brush_opacity": int(self.brush_opacity),
            "brush_spacing": int(self.brush_spacing),
        }

    @classmethod
    def from_dict(cls, data: object) -> Optional["StrokeRecord"]:
        if not isinstance(data, dict):
            return None
        raw_points = data.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            return None
        points: List[Tuple[float, float]] = []
        for item in raw_points:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                points.append((float(item[0]), float(item[1])))
            except Exception:
                continue
        if not points:
            return None
        return cls(
            erase=bool(data.get("erase", False)),
            points=points,
            brush_size=clamp_int(int(data.get("brush_size", 1)), *PARAM_RANGES["brush_size"]),
            brush_hardness=clamp_int(int(data.get("brush_hardness", 0)), *PARAM_RANGES["brush_hardness"]),
            brush_opacity=clamp_int(int(data.get("brush_opacity", 1)), *PARAM_RANGES["brush_opacity"]),
            brush_spacing=clamp_int(int(data.get("brush_spacing", 1)), *PARAM_RANGES["brush_spacing"]),
        )


def clone_stroke_records(records: Iterable[StrokeRecord]) -> List[StrokeRecord]:
    return [StrokeRecord.from_dict(record.to_dict()) for record in records if StrokeRecord.from_dict(record.to_dict()) is not None]


def encode_rgba_png_base64(rgba: np.ndarray) -> str:
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode('.png', bgra)
    if not ok:
        raise ValueError('画像の埋め込みエンコードに失敗しました。')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def decode_rgba_png_base64(text_b64: str) -> np.ndarray:
    data = base64.b64decode(text_b64.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('埋め込み画像のデコードに失敗しました。')
    if img.ndim == 2:
        rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return rgba


def replay_mask_from_strokes(shape: Tuple[int, int], strokes: Iterable[StrokeRecord]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    stamp_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
    for stroke in strokes:
        points = stroke.points
        if not points:
            continue
        stamp_key = (stroke.brush_size, stroke.brush_hardness, stroke.brush_opacity)
        stamp = stamp_cache.get(stamp_key)
        if stamp is None:
            stamp = create_brush_stamp(*stamp_key)
            stamp_cache[stamp_key] = stamp
        spacing_px = max(1.0, stroke.brush_size * stroke.brush_spacing / 100.0)
        blend_stamp(mask, stamp, points[0][0], points[0][1], stroke.erase)
        last_x, last_y = points[0]
        for x, y in points[1:]:
            dx = x - last_x
            dy = y - last_y
            dist = math.hypot(dx, dy)
            steps = max(1, int(dist / spacing_px))
            for i in range(1, steps + 1):
                t = i / steps
                px = last_x + dx * t
                py = last_y + dy * t
                blend_stamp(mask, stamp, px, py, stroke.erase)
            last_x, last_y = x, y
    return mask


def build_center_line_mask(shape: Tuple[int, int], strokes: Iterable[StrokeRecord], width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(width))
    for stroke in strokes:
        pts = stroke.points
        if not pts:
            continue
        points = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
        color = 0 if stroke.erase else 255
        if len(points) == 1:
            cv2.circle(mask, tuple(points[0]), max(1, thickness // 2), color, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(mask, [points.reshape((-1, 1, 2))], False, color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.circle(mask, tuple(points[0]), max(1, thickness // 2), color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(mask, tuple(points[-1]), max(1, thickness // 2), color, thickness=-1, lineType=cv2.LINE_AA)
    return mask


PARAM_RANGES: Dict[str, Tuple[int, int]] = {
    "brush_size": (1, 300),
    "brush_hardness": (0, 100),
    "brush_opacity": (1, 100),
    "brush_spacing": (1, 100),
    "smoothing": (0, 100),
    "depth": (0, 100),
    "displacement": (0, 50),
    "pull": (0, 100),
    "rim": (0, 100),
    "edge_blur": (0, 50),
    "texture_noise": (0, 100),
    "shadow_strength": (0, 100),
    "highlight_strength": (0, 100),
    "light_angle": (0, 360),
    "shadow_blur": (0, 50),
    "shadow_spread": (0, 100),
    "highlight_width": (0, 50),
    "inner_dark": (0, 100),
    "final_blur_enabled": (0, 1),
    "final_blur_size": (0, 30),
    "final_blur_strength": (0, 100),
    "center_line_enabled": (0, 1),
    "center_line_width": (0, 30),
    "center_line_opacity": (0, 100),
    "jpeg_quality": (1, 100),
    "webp_quality": (1, 100),
}

PRESETS: Dict[str, Dict[str, int]] = {
    "弱め": {
        "brush_hardness": 25,
        "brush_opacity": 35,
        "depth": 14,
        "displacement": 3,
        "pull": 24,
        "rim": 8,
        "edge_blur": 12,
        "shadow_strength": 22,
        "highlight_strength": 12,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
        "inner_dark": 10,
    },
    "標準": {
        "brush_size": 5,
        "brush_hardness": 62,
        "brush_opacity": 72,
        "brush_spacing": 15,
        "smoothing": 24,
        "depth": 40,
        "displacement": 34,
        "pull": 41,
        "rim": 38,
        "edge_blur": 8,
        "texture_noise": 14,
        "shadow_strength": 10,
        "highlight_strength": 0,
        "light_angle": 14,
        "shadow_blur": 6,
        "shadow_spread": 39,
        "highlight_width": 10,
        "inner_dark": 50,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
    },
    "強め": {
        "brush_hardness": 45,
        "brush_opacity": 70,
        "depth": 45,
        "displacement": 12,
        "pull": 55,
        "rim": 25,
        "edge_blur": 7,
        "shadow_strength": 55,
        "highlight_strength": 38,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
        "inner_dark": 36,
    },
    "細い食い込み": {
        "brush_size": 18,
        "brush_hardness": 62,
        "brush_opacity": 72,
        "depth": 42,
        "displacement": 8,
        "pull": 66,
        "rim": 28,
        "edge_blur": 4,
        "shadow_strength": 54,
        "highlight_strength": 42,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
        "inner_dark": 35,
    },
    "柔らかい押し跡": {
        "brush_size": 80,
        "brush_hardness": 15,
        "brush_opacity": 45,
        "depth": 22,
        "displacement": 5,
        "pull": 25,
        "rim": 10,
        "edge_blur": 20,
        "shadow_strength": 26,
        "highlight_strength": 15,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
        "inner_dark": 16,
    },
    "硬い刻み": {
        "brush_size": 24,
        "brush_hardness": 85,
        "brush_opacity": 86,
        "depth": 55,
        "displacement": 10,
        "pull": 70,
        "rim": 42,
        "edge_blur": 2,
        "shadow_strength": 66,
        "highlight_strength": 54,
        "final_blur_enabled": 1,
        "final_blur_size": 0,
        "final_blur_strength": 0,
        "center_line_enabled": 1,
        "center_line_width": 0,
        "center_line_opacity": 100,
        "inner_dark": 45,
    },
}


PRESET_PARAM_KEYS: Tuple[str, ...] = tuple(
    key for key in PARAM_RANGES.keys() if key not in {"jpeg_quality", "webp_quality"}
)


def sanitize_presets(data: object, fallback: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Dict[str, int]]:
    source = data if isinstance(data, dict) else fallback
    if not isinstance(source, dict):
        source = PRESETS
    cleaned: Dict[str, Dict[str, int]] = {}
    for raw_name, raw_values in source.items():
        name = str(raw_name).strip()
        if not name or not isinstance(raw_values, dict):
            continue
        preset: Dict[str, int] = {}
        for key, value in raw_values.items():
            if key not in PRESET_PARAM_KEYS:
                continue
            lo, hi = PARAM_RANGES[key]
            try:
                preset[key] = clamp_int(int(value), lo, hi)
            except Exception:
                continue
        if preset:
            cleaned[name] = preset
    if not cleaned:
        cleaned = {name: dict(values) for name, values in PRESETS.items()}
    return cleaned


def current_params_as_preset(params: ToolParams) -> Dict[str, int]:
    return {key: int(getattr(params, key)) for key in PRESET_PARAM_KEYS}


def unique_preset_name(base: str, existing: Iterable[str]) -> str:
    name = base.strip() or "新規プリセット"
    existing_set = set(existing)
    if name not in existing_set:
        return name
    index = 2
    while f"{name} {index}" in existing_set:
        index += 1
    return f"{name} {index}"


def normalize_hex_color(value: object, default: str = DEFAULT_CENTER_LINE_COLOR) -> str:
    text = str(value).strip()
    if not text:
        return default
    if not text.startswith("#"):
        text = f"#{text}"
    if len(text) != 7:
        return default
    try:
        int(text[1:], 16)
    except Exception:
        return default
    return text.upper()


def skeletonize_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    img = (binary_mask > 0).astype(np.uint8) * 255
    if img.size == 0 or not np.any(img):
        return np.zeros_like(img)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def gaussian_kernel_size(radius: int) -> int:
    radius = max(0, int(radius))
    if radius <= 0:
        return 0
    k = radius * 2 + 1
    return k if k % 2 == 1 else k + 1


def create_brush_stamp(size: int, hardness: int, opacity: int) -> np.ndarray:
    size = max(1, int(size))
    radius = max(0.5, size / 2.0)
    dim = int(math.ceil(radius * 2)) + 2
    cy = cx = dim / 2.0 - 0.5
    y, x = np.ogrid[:dim, :dim]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    hard = max(0.0, min(1.0, hardness / 100.0))
    inner = radius * hard
    feather = max(0.001, radius - inner)
    alpha = np.ones((dim, dim), dtype=np.float32)
    alpha[dist > radius] = 0.0
    edge = (dist - inner) / feather
    feathered = 1.0 - np.clip(edge, 0.0, 1.0)
    alpha = np.where(dist <= inner, 1.0, feathered)
    alpha *= max(0.0, min(1.0, opacity / 100.0)) * 255.0
    return np.clip(alpha, 0, 255).astype(np.uint8)


def blend_stamp(mask: np.ndarray, stamp: np.ndarray, cx: float, cy: float, erase: bool) -> None:
    h, w = mask.shape[:2]
    sh, sw = stamp.shape[:2]
    x0 = int(round(cx - sw / 2))
    y0 = int(round(cy - sh / 2))
    x1 = x0 + sw
    y1 = y0 + sh
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(w, x1)
    iy1 = min(h, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    sx0 = ix0 - x0
    sy0 = iy0 - y0
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)
    sub = mask[iy0:iy1, ix0:ix1]
    alpha = stamp[sy0:sy1, sx0:sx1]
    if erase:
        reduced = sub.astype(np.int16) - alpha.astype(np.int16)
        mask[iy0:iy1, ix0:ix1] = np.clip(reduced, 0, 255).astype(np.uint8)
    else:
        mask[iy0:iy1, ix0:ix1] = np.maximum(sub, alpha)


def apply_dent_effect(original_rgba: np.ndarray, mask_u8: np.ndarray, params: ToolParams, center_line_color: str = DEFAULT_CENTER_LINE_COLOR, stroke_records: Optional[Iterable[StrokeRecord]] = None) -> np.ndarray:
    if original_rgba is None or mask_u8 is None:
        return original_rgba
    if original_rgba.size == 0:
        return original_rgba

    h, w = mask_u8.shape[:2]
    src = original_rgba.astype(np.float32)
    rgb = src[:, :, :3]
    alpha = src[:, :, 3]

    mask = mask_u8.astype(np.float32) / 255.0
    if params.edge_blur > 0:
        k = gaussian_kernel_size(params.edge_blur)
        if k > 0:
            mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
        else:
            mask_blur = mask
    else:
        mask_blur = mask
    mask_blur = np.clip(mask_blur, 0.0, 1.0)

    if np.max(mask_blur) <= 0.0001:
        return original_rgba.copy()

    # Height map: center is lower. Edge and gradients are used for pseudo 3D shading.
    depth_scale = params.depth / 100.0
    height = -mask_blur * depth_scale

    grad_x = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)

    disp_amount = float(params.displacement) * (params.pull / 100.0)
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x_grid - grad_x * disp_amount * 32.0
    map_y = y_grid - grad_y * disp_amount * 32.0

    if params.rim > 0:
        rim_strength = params.rim / 100.0
        lap = cv2.Laplacian(mask_blur, cv2.CV_32F, ksize=3)
        map_x += grad_x * lap * rim_strength * 20.0
        map_y += grad_y * lap * rim_strength * 20.0

    warped_rgb = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_alpha = cv2.remap(
        alpha,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    if warped_alpha.ndim == 3:
        warped_alpha = warped_alpha[:, :, 0]

    nx = -grad_x * (1.0 + depth_scale * 10.0)
    ny = -grad_y * (1.0 + depth_scale * 10.0)
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= norm
    ny /= norm
    nz /= norm

    angle = math.radians(params.light_angle)
    lx = math.cos(angle)
    ly = math.sin(angle)
    lz = 0.65
    lnorm = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / lnorm, ly / lnorm, lz / lnorm

    dot = nx * lx + ny * ly + nz * lz
    baseline = lz
    directional = dot - baseline

    shadow = np.clip(-directional, 0.0, 1.0) * (params.shadow_strength / 100.0)
    highlight = np.clip(directional, 0.0, 1.0) * (params.highlight_strength / 100.0)

    if params.shadow_blur > 0:
        k = gaussian_kernel_size(params.shadow_blur)
        if k > 0:
            shadow = cv2.GaussianBlur(shadow, (k, k), 0)
    if params.shadow_spread > 0:
        spread_k = gaussian_kernel_size(max(1, int(params.shadow_spread / 10)))
        if spread_k > 0:
            shadow = cv2.dilate(shadow, np.ones((spread_k, spread_k), np.uint8), iterations=1)
            shadow = cv2.GaussianBlur(shadow, (spread_k, spread_k), 0)
    if params.highlight_width > 0:
        k = gaussian_kernel_size(max(1, int(params.highlight_width / 2)))
        if k > 0:
            highlight = cv2.GaussianBlur(highlight, (k, k), 0)

    inner = mask_blur * (params.inner_dark / 100.0) * 0.45
    shadow = np.clip(shadow + inner, 0.0, 1.0) * mask_blur
    highlight = np.clip(highlight, 0.0, 1.0) * mask_blur

    if params.texture_noise > 0:
        rng = np.random.default_rng(12345)
        noise = rng.normal(0.0, 0.035 * (params.texture_noise / 100.0), size=mask_blur.shape).astype(np.float32)
        shadow = np.clip(shadow + noise * mask_blur, 0.0, 1.0)
        highlight = np.clip(highlight - noise * mask_blur * 0.5, 0.0, 1.0)

    shaded = warped_rgb.copy()
    shaded *= (1.0 - shadow[:, :, None] * 0.95)
    shaded += (255.0 - shaded) * (highlight[:, :, None] * 0.75)
    shaded = np.clip(shaded, 0, 255)

    blend_rgb = mask_blur[:, :, None]
    blend_alpha = mask_blur
    out_rgb = rgb * (1.0 - blend_rgb) + shaded * blend_rgb
    out_alpha = alpha * (1.0 - blend_alpha) + warped_alpha * blend_alpha
    out = np.dstack([out_rgb, out_alpha])
    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)

    if params.center_line_enabled and params.center_line_width > 0 and params.center_line_opacity > 0 and stroke_records is not None:
        line_mask = build_center_line_mask(mask_u8.shape[:2], stroke_records, params.center_line_width)
        if np.any(line_mask):
            line_blend = (line_mask.astype(np.float32) / 255.0) * (params.center_line_opacity / 100.0)
            if np.any(line_blend > 0):
                line_color = normalize_hex_color(center_line_color)
                qr = int(line_color[1:3], 16)
                qg = int(line_color[3:5], 16)
                qb = int(line_color[5:7], 16)
                color_arr = np.array([qr, qg, qb], dtype=np.float32).reshape((1, 1, 3))
                out_rgb_f = out[:, :, :3].astype(np.float32)
                out_rgb_f = out_rgb_f * (1.0 - line_blend[:, :, None]) + color_arr * line_blend[:, :, None]
                out[:, :, :3] = np.rint(np.clip(out_rgb_f, 0, 255)).astype(np.uint8)

    if params.final_blur_enabled and params.final_blur_size > 0 and params.final_blur_strength > 0:
        blur_radius = int(params.final_blur_size)
        blend_amount = params.final_blur_strength / 100.0
        if blur_radius > 0 and blend_amount > 0.0:
            region = (mask_u8 > 0).astype(np.uint8)
            if np.any(region):
                expand_radius = blur_radius
                kernel_dim = expand_radius * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim))
                expanded_region = cv2.dilate(region, kernel, iterations=1).astype(np.float32)
                blur_falloff_k = gaussian_kernel_size(max(1, blur_radius))
                soft_region = cv2.GaussianBlur(expanded_region, (blur_falloff_k, blur_falloff_k), 0)
                soft_region = np.clip(soft_region, 0.0, 1.0)
                blur_k = gaussian_kernel_size(blur_radius)
                if blur_k > 0:
                    blurred_rgb = cv2.GaussianBlur(out[:, :, :3].astype(np.float32), (blur_k, blur_k), 0)
                    blurred_alpha = cv2.GaussianBlur(out[:, :, 3].astype(np.float32), (blur_k, blur_k), 0)
                    blend_mask = soft_region * blend_amount
                    out_rgb_f = out[:, :, :3].astype(np.float32)
                    out_alpha_f = out[:, :, 3].astype(np.float32)
                    out_rgb_f = out_rgb_f * (1.0 - blend_mask[:, :, None]) + blurred_rgb * blend_mask[:, :, None]
                    out_alpha_f = out_alpha_f * (1.0 - blend_mask) + blurred_alpha * blend_mask
                    out = np.dstack([out_rgb_f, out_alpha_f])
                    out = np.rint(np.clip(out, 0, 255)).astype(np.uint8)

    return out


class ParamControl(QWidget):
    valueChanged = Signal(str, int)

    def __init__(self, key: str, label: str, lo: int, hi: int, value: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.key = key
        self._block = False
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(value)
        self.spin.setFixedWidth(74)
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.slider, 0, 1)
        layout.addWidget(self.spin, 0, 2)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_changed(self, value: int) -> None:
        if self._block:
            return
        self._block = True
        self.spin.setValue(value)
        self._block = False
        self.valueChanged.emit(self.key, value)

    def _spin_changed(self, value: int) -> None:
        if self._block:
            return
        self._block = True
        self.slider.setValue(value)
        self._block = False
        self.valueChanged.emit(self.key, value)

    def set_value(self, value: int) -> None:
        self._block = True
        self.slider.setValue(value)
        self.spin.setValue(value)
        self._block = False


class ImageCanvas(QWidget):
    strokeStarted = Signal()
    strokePainted = Signal(float, float, bool)
    strokeFinished = Signal()
    zoomChanged = Signal(float)
    contextMenuRequestedAt = Signal(QPoint)
    fileDropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.qimage: Optional[QImage] = None
        self.pixmap: Optional[QPixmap] = None
        self.display_rgba: Optional[np.ndarray] = None
        self._scaled_cache_key: Optional[Tuple[int, int, int]] = None
        self._scaled_cache_pixmap: Optional[QPixmap] = None
        self.image_size = QSize(0, 0)
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.dragging = False
        self.panning = False
        self.last_mouse = QPointF(0, 0)
        self.cursor_img_pos: Optional[QPointF] = None
        self.brush_size = 40
        self.tool = "brush"
        self.space_down = False
        self.before_after_active = False
        self._pan_moved = False
        self.live_stroke_points: List[QPointF] = []
        self.live_stroke_erase = False
        self.setMinimumSize(420, 420)

    def set_image(self, qimage: Optional[QImage]) -> None:
        self.qimage = qimage
        self.pixmap = QPixmap.fromImage(qimage) if qimage is not None else None
        self.display_rgba = qimage_to_rgba(qimage) if qimage is not None else None
        self._clear_scaled_cache()
        if qimage is not None:
            self.image_size = QSize(qimage.width(), qimage.height())
            self.fit_to_window()
        else:
            self.image_size = QSize(0, 0)
            self.update()

    def replace_image_without_fit(self, qimage: Optional[QImage]) -> None:
        self.qimage = qimage
        self.pixmap = QPixmap.fromImage(qimage) if qimage is not None else None
        self.display_rgba = qimage_to_rgba(qimage) if qimage is not None else None
        self._clear_scaled_cache()
        if qimage is not None:
            self.image_size = QSize(qimage.width(), qimage.height())
        self.update()

    def _clear_scaled_cache(self) -> None:
        self._scaled_cache_key = None
        self._scaled_cache_pixmap = None

    def _bicubic_scaled_pixmap(self, width: int, height: int) -> Optional[QPixmap]:
        if self.display_rgba is None:
            return None
        width = max(1, int(width))
        height = max(1, int(height))
        cache_key = (id(self.display_rgba), width, height)
        if self._scaled_cache_key == cache_key and self._scaled_cache_pixmap is not None:
            return self._scaled_cache_pixmap
        # Avoid creating very large temporary pixmaps while zoomed far in.
        # Normal zoom levels use OpenCV INTER_CUBIC for better preview quality.
        if width * height > 45_000_000:
            return None
        scaled = cv2.resize(self.display_rgba, (width, height), interpolation=cv2.INTER_CUBIC)
        pixmap = QPixmap.fromImage(ndarray_rgba_to_qimage(scaled))
        self._scaled_cache_key = cache_key
        self._scaled_cache_pixmap = pixmap
        return pixmap

    def fit_to_window(self) -> None:
        if self.image_size.width() <= 0 or self.image_size.height() <= 0:
            return
        margin = 30
        zw = max(0.01, (self.width() - margin) / self.image_size.width())
        zh = max(0.01, (self.height() - margin) / self.image_size.height())
        self.zoom = min(zw, zh, 1.0)
        draw_w = self.image_size.width() * self.zoom
        draw_h = self.image_size.height() * self.zoom
        self.offset = QPointF((self.width() - draw_w) / 2.0, (self.height() - draw_h) / 2.0)
        self.zoomChanged.emit(self.zoom)
        self.update()

    def set_actual_size(self) -> None:
        if self.image_size.width() <= 0:
            return
        center_img = self.view_to_image(QPointF(self.width() / 2, self.height() / 2))
        self.zoom = 1.0
        self.offset = QPointF(self.width() / 2 - center_img.x() * self.zoom, self.height() / 2 - center_img.y() * self.zoom)
        self.zoomChanged.emit(self.zoom)
        self.update()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = size
        self.update()

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self.update()

    def begin_live_stroke(self, erase: bool) -> None:
        self.live_stroke_points = []
        self.live_stroke_erase = erase
        self.update()

    def add_live_stroke_point(self, x: float, y: float, erase: bool) -> None:
        point = QPointF(float(x), float(y))
        self.live_stroke_erase = erase
        if self.live_stroke_points:
            last = self.live_stroke_points[-1]
            if math.hypot(point.x() - last.x(), point.y() - last.y()) < 0.5:
                return
        self.live_stroke_points.append(point)
        self.update()

    def end_live_stroke(self) -> None:
        self.live_stroke_points = []
        self.update()


    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in SUPPORTED_EXTS:
                        self.fileDropped.emit(str(path))
                        event.acceptProposedAction()
                        return
        event.ignore()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(38, 38, 42))
        if self.pixmap is None:
            painter.setPen(QColor(190, 190, 190))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "画像をD&D、または ファイル > 開く")
            return

        target = QRectF(
            self.offset.x(),
            self.offset.y(),
            self.image_size.width() * self.zoom,
            self.image_size.height() * self.zoom,
        )
        target_w = max(1, int(round(self.image_size.width() * self.zoom)))
        target_h = max(1, int(round(self.image_size.height() * self.zoom)))
        scaled_pixmap = None
        if self.zoom != 1.0:
            scaled_pixmap = self._bicubic_scaled_pixmap(target_w, target_h)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if scaled_pixmap is not None:
            scaled_target = QRectF(self.offset.x(), self.offset.y(), target_w, target_h)
            painter.drawPixmap(scaled_target, scaled_pixmap, QRectF(0, 0, target_w, target_h))
        else:
            # Fallback for 100% display or extremely large zoomed views.
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom != 1.0)
            painter.drawPixmap(target, self.pixmap, QRectF(0, 0, self.image_size.width(), self.image_size.height()))
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        painter.setPen(QPen(QColor(90, 90, 95), 1))
        painter.drawRect(target)

        if self.live_stroke_points:
            painter.save()
            painter.setClipRect(target)
            overlay_color = QColor(120, 210, 255, 135) if self.live_stroke_erase else QColor(255, 220, 80, 125)
            width = max(1.0, self.brush_size * self.zoom)
            pen = QPen(overlay_color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            if len(self.live_stroke_points) == 1:
                c = self.image_to_view(self.live_stroke_points[0])
                painter.setBrush(overlay_color)
                painter.drawEllipse(c, width / 2.0, width / 2.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                path = QPainterPath(self.image_to_view(self.live_stroke_points[0]))
                for point in self.live_stroke_points[1:]:
                    path.lineTo(self.image_to_view(point))
                painter.drawPath(path)
            painter.restore()

        if self.cursor_img_pos is not None and self.tool in {"brush", "eraser"}:
            c = self.image_to_view(self.cursor_img_pos)
            r = max(1.0, self.brush_size * self.zoom / 2.0)
            color = QColor(255, 230, 120) if self.tool == "brush" else QColor(120, 210, 255)
            pen = QPen(color, 1.5)
            painter.setPen(pen)
            painter.drawEllipse(c, r, r)
            painter.setPen(QPen(QColor(0, 0, 0, 150), 1))
            painter.drawEllipse(c, max(1.0, r - 1.5), max(1.0, r - 1.5))

    def image_to_view(self, p: QPointF) -> QPointF:
        return QPointF(self.offset.x() + p.x() * self.zoom, self.offset.y() + p.y() * self.zoom)

    def view_to_image(self, p: QPointF) -> QPointF:
        if self.zoom <= 0:
            return QPointF(0, 0)
        return QPointF((p.x() - self.offset.x()) / self.zoom, (p.y() - self.offset.y()) / self.zoom)

    def point_inside_image(self, img_p: QPointF) -> bool:
        return 0 <= img_p.x() < self.image_size.width() and 0 <= img_p.y() < self.image_size.height()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = QPointF(event.position())
        self.last_mouse = pos
        img_p = self.view_to_image(pos)
        if event.button() == Qt.MouseButton.RightButton or self.space_down:
            self.panning = True
            self._pan_moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self.tool in {"brush", "eraser"} and self.point_inside_image(img_p):
            self.dragging = True
            self.strokeStarted.emit()
            self.strokePainted.emit(img_p.x(), img_p.y(), self.tool == "eraser")
            return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        pos = QPointF(event.position())
        img_p = self.view_to_image(pos)
        self.cursor_img_pos = img_p if self.point_inside_image(img_p) else None
        if self.dragging:
            self.strokePainted.emit(img_p.x(), img_p.y(), self.tool == "eraser")
        elif self.panning:
            delta = pos - self.last_mouse
            if abs(delta.x()) + abs(delta.y()) > 0.5:
                self._pan_moved = True
            self.offset += delta
            self.last_mouse = pos
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            self.strokeFinished.emit()
        if event.button() == Qt.MouseButton.RightButton and self.panning:
            should_menu = not self._pan_moved
            self.panning = False
            self.unsetCursor()
            if should_menu:
                self.contextMenuRequestedAt.emit(event.globalPosition().toPoint())

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.cursor_img_pos = None
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self.pixmap is None:
            return
        pos = QPointF(event.position())
        before = self.view_to_image(pos)
        steps = event.angleDelta().y() / 120.0
        factor = 1.15 ** steps
        self.zoom = max(0.02, min(16.0, self.zoom * factor))
        self.offset = QPointF(pos.x() - before.x() * self.zoom, pos.y() - before.y() * self.zoom)
        self.zoomChanged.emit(self.zoom)
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self.space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self.space_down = False
            if not self.panning:
                self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.params = ToolParams()
        self.param_controls: Dict[str, ParamControl] = {}
        self.original_rgba: Optional[np.ndarray] = None
        self.preview_rgba: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.image_path: Optional[Path] = None
        self.image_display_name: str = "未読み込み"
        self.project_path: Optional[Path] = None
        self.undo_stack: List[Dict[str, object]] = []
        self.redo_stack: List[Dict[str, object]] = []
        self.max_undo = 50
        self.stroke_history: List[StrokeRecord] = []
        self._stroke_active = False
        self._stroke_before_state: Optional[Dict[str, object]] = None
        self._current_stroke_points: List[Tuple[float, float]] = []
        self._current_stroke_erase = False
        self._last_paint: Optional[Tuple[float, float]] = None
        self._smoothed_paint: Optional[Tuple[float, float]] = None
        self._render_pending = False
        self._show_before = False
        self._settings_warning = ""
        self._brush_stamp_cache_key: Optional[Tuple[int, int, int]] = None
        self._brush_stamp_cache: Optional[np.ndarray] = None
        self.presets: Dict[str, Dict[str, int]] = sanitize_presets(PRESETS)
        self.current_preset_name = "標準" if "標準" in self.presets else next(iter(self.presets))
        self.center_line_color = DEFAULT_CENTER_LINE_COLOR

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setAcceptDrops(True)
        self.resize(1180, 780)
        self._build_ui()
        self._build_actions()
        self._load_settings()
        if self._settings_warning:
            self.statusBar().showMessage(self._settings_warning, 7000)
        else:
            self.statusBar().showMessage("画像をD&D、または ファイル > 開く")

    def _build_ui(self) -> None:
        self.canvas = ImageCanvas()
        self.canvas.strokeStarted.connect(self.on_stroke_started)
        self.canvas.strokePainted.connect(self.on_stroke_painted)
        self.canvas.strokeFinished.connect(self.on_stroke_finished)
        self.canvas.zoomChanged.connect(self.on_zoom_changed)
        self.canvas.contextMenuRequestedAt.connect(self.show_canvas_context_menu)
        self.canvas.fileDropped.connect(lambda p: self.load_image_path(Path(p)))

        self.sidebar = QWidget()
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)

        tool_group = QGroupBox("ツール")
        tool_layout = QGridLayout(tool_group)
        self.brush_btn = QToolButton()
        self.brush_btn.setText("凹みブラシ")
        self.brush_btn.setCheckable(True)
        self.brush_btn.setChecked(True)
        self.eraser_btn = QToolButton()
        self.eraser_btn.setText("消しゴム")
        self.eraser_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self.set_tool("brush"))
        self.eraser_btn.clicked.connect(lambda: self.set_tool("eraser"))
        tool_layout.addWidget(self.brush_btn, 0, 0)
        tool_layout.addWidget(self.eraser_btn, 0, 1)
        side_layout.addWidget(tool_group)

        preset_group = QGroupBox("プリセット")
        preset_layout = QHBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preset_combo.activated.connect(self.on_preset_activated)
        line_edit = self.preset_combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self.on_preset_name_editing_finished)
        self.new_preset_btn = QPushButton("New")
        self.new_preset_btn.clicked.connect(self.new_preset_from_current)
        self.del_preset_btn = QPushButton("Del")
        self.del_preset_btn.clicked.connect(self.delete_current_preset)
        preset_layout.addWidget(self.preset_combo, 1)
        preset_layout.addWidget(self.new_preset_btn)
        preset_layout.addWidget(self.del_preset_btn)
        side_layout.addWidget(preset_group)
        self.refresh_preset_ui(self.current_preset_name)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_host = QWidget()
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._add_group(
            controls_layout,
            "ブラシ",
            [
                ("brush_size", "サイズ", "brush_size"),
                ("brush_hardness", "硬さ", "brush_hardness"),
                ("brush_opacity", "適用量", "brush_opacity"),
                ("brush_spacing", "間隔", "brush_spacing"),
                ("smoothing", "手ぶれ補正", "smoothing"),
            ],
        )
        self._add_group(
            controls_layout,
            "凹み",
            [
                ("depth", "凹み深さ", "depth"),
                ("displacement", "変位量", "displacement"),
                ("pull", "中心への引き込み", "pull"),
                ("rim", "縁の盛り上がり", "rim"),
                ("edge_blur", "境界ぼかし", "edge_blur"),
                ("texture_noise", "質感ノイズ", "texture_noise"),
            ],
        )
        self._add_group(
            controls_layout,
            "陰影",
            [
                ("shadow_strength", "影の強さ", "shadow_strength"),
                ("highlight_strength", "ハイライト", "highlight_strength"),
                ("light_angle", "光源角度", "light_angle"),
                ("shadow_blur", "影ぼかし", "shadow_blur"),
                ("shadow_spread", "影の広がり", "shadow_spread"),
                ("highlight_width", "ハイライト幅", "highlight_width"),
                ("inner_dark", "内部の暗さ", "inner_dark"),
            ],
        )
        center_line_group = QGroupBox("中心線")
        center_line_layout = QVBoxLayout(center_line_group)
        center_line_layout.setContentsMargins(8, 8, 8, 8)
        self.center_line_enabled_check = QCheckBox("有効")
        self.center_line_enabled_check.setChecked(bool(self.params.center_line_enabled))
        self.center_line_enabled_check.toggled.connect(self.on_center_line_enabled_toggled)
        center_line_layout.addWidget(self.center_line_enabled_check)
        line_width_control = ParamControl(
            "center_line_width",
            "太さ",
            PARAM_RANGES["center_line_width"][0],
            PARAM_RANGES["center_line_width"][1],
            getattr(self.params, "center_line_width"),
        )
        line_width_control.valueChanged.connect(self.on_param_changed)
        self.param_controls["center_line_width"] = line_width_control
        center_line_layout.addWidget(line_width_control)
        line_opacity_control = ParamControl(
            "center_line_opacity",
            "不透明度",
            PARAM_RANGES["center_line_opacity"][0],
            PARAM_RANGES["center_line_opacity"][1],
            getattr(self.params, "center_line_opacity"),
        )
        line_opacity_control.valueChanged.connect(self.on_param_changed)
        self.param_controls["center_line_opacity"] = line_opacity_control
        center_line_layout.addWidget(line_opacity_control)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("色"))
        self.center_line_color_edit = QLineEdit(self.center_line_color)
        self.center_line_color_edit.setPlaceholderText("#RRGGBB")
        self.center_line_color_edit.editingFinished.connect(self.on_center_line_color_editing_finished)
        color_row.addWidget(self.center_line_color_edit, 1)
        self.center_line_color_btn = QPushButton("...")
        self.center_line_color_btn.setFixedWidth(42)
        self.center_line_color_btn.clicked.connect(self.choose_center_line_color)
        color_row.addWidget(self.center_line_color_btn)
        center_line_layout.addLayout(color_row)
        controls_layout.addWidget(center_line_group)
        self.center_line_group = center_line_group
        self._update_center_line_color_widgets()
        self._update_effect_group_enabled_states()

        final_blur_group = QGroupBox("最終ぼかし")
        final_blur_layout = QVBoxLayout(final_blur_group)
        final_blur_layout.setContentsMargins(8, 8, 8, 8)
        self.final_blur_enabled_check = QCheckBox("有効")
        self.final_blur_enabled_check.setChecked(bool(self.params.final_blur_enabled))
        self.final_blur_enabled_check.toggled.connect(self.on_final_blur_enabled_toggled)
        final_blur_layout.addWidget(self.final_blur_enabled_check)
        final_blur_size_control = ParamControl(
            "final_blur_size",
            "サイズ",
            PARAM_RANGES["final_blur_size"][0],
            PARAM_RANGES["final_blur_size"][1],
            getattr(self.params, "final_blur_size"),
        )
        final_blur_size_control.valueChanged.connect(self.on_param_changed)
        self.param_controls["final_blur_size"] = final_blur_size_control
        final_blur_layout.addWidget(final_blur_size_control)
        final_blur_strength_control = ParamControl(
            "final_blur_strength",
            "強度",
            PARAM_RANGES["final_blur_strength"][0],
            PARAM_RANGES["final_blur_strength"][1],
            getattr(self.params, "final_blur_strength"),
        )
        final_blur_strength_control.valueChanged.connect(self.on_param_changed)
        self.param_controls["final_blur_strength"] = final_blur_strength_control
        final_blur_layout.addWidget(final_blur_strength_control)
        controls_layout.addWidget(final_blur_group)
        self.final_blur_group = final_blur_group

        self._add_group(
            controls_layout,
            "保存品質",
            [
                ("jpeg_quality", "JPEG品質", "jpeg_quality"),
                ("webp_quality", "WEBP品質", "webp_quality"),
            ],
        )
        controls_layout.addStretch(1)
        scroll.setWidget(controls_host)
        side_layout.addWidget(scroll, 1)

        self.info_label = QLabel("未読み込み")
        self.info_label.setWordWrap(True)
        side_layout.addWidget(self.info_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([860, 320])
        self.splitter = splitter
        self.setCentralWidget(splitter)

        self.statusBar().show()

    def _add_group(self, parent_layout: QVBoxLayout, title: str, rows: Iterable[Tuple[str, str, str]]) -> None:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        for key, label, attr in rows:
            lo, hi = PARAM_RANGES[attr]
            control = ParamControl(attr, label, lo, hi, getattr(self.params, attr))
            control.valueChanged.connect(self.on_param_changed)
            self.param_controls[attr] = control
            layout.addWidget(control)
        parent_layout.addWidget(group)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")
        edit_menu = self.menuBar().addMenu("編集")
        view_menu = self.menuBar().addMenu("表示")
        tool_menu = self.menuBar().addMenu("ツール")
        self.preset_menu = self.menuBar().addMenu("プリセット")

        self.open_action = QAction("画像を開く", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image_dialog)
        file_menu.addAction(self.open_action)

        self.open_project_action = QAction("プロジェクトを開く", self)
        self.open_project_action.triggered.connect(self.open_project_dialog)
        file_menu.addAction(self.open_project_action)

        self.paste_action = QAction("クリップボードから貼り付け", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.triggered.connect(self.paste_from_clipboard)
        file_menu.addAction(self.paste_action)

        file_menu.addSeparator()
        self.save_project_action = QAction("プロジェクトを保存", self)
        self.save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(self.save_project_action)

        self.save_project_as_action = QAction("プロジェクトに名前を付けて保存", self)
        self.save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(self.save_project_as_action)

        file_menu.addSeparator()
        self.save_action = QAction("画像を書き出し", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self.save_image)
        file_menu.addAction(self.save_action)

        self.save_as_action = QAction("画像に名前を付けて保存", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(self.save_as_action)

        file_menu.addSeparator()
        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.undo_action = QAction("元に戻す", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("やり直し", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()
        clear_action = QAction("全ての凹みをクリア", self)
        clear_action.triggered.connect(self.clear_all_mask_confirm)
        edit_menu.addAction(clear_action)

        fit_action = QAction("全体表示", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)

        actual_action = QAction("100%表示", self)
        actual_action.setShortcut("Ctrl+1")
        actual_action.triggered.connect(self.canvas.set_actual_size)
        view_menu.addAction(actual_action)

        before_action = QAction("Before / After", self)
        before_action.setShortcut("Tab")
        before_action.triggered.connect(self.toggle_before_after)
        view_menu.addAction(before_action)

        brush_action = QAction("凹みブラシ", self)
        brush_action.setShortcut("1")
        brush_action.triggered.connect(lambda: self.set_tool("brush"))
        tool_menu.addAction(brush_action)

        eraser_action = QAction("消しゴム", self)
        eraser_action.setShortcut("2")
        eraser_action.triggered.connect(lambda: self.set_tool("eraser"))
        tool_menu.addAction(eraser_action)

        self.rebuild_preset_menu()

        self._update_action_states()

    def _load_settings(self) -> None:
        path = settings_path()
        if not path.exists():
            self._sync_param_controls()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("params"), dict):
                self.params = ToolParams.from_dict(data["params"])
            if isinstance(data.get("presets"), dict):
                self.presets = sanitize_presets(data.get("presets"), PRESETS)
                self.current_preset_name = str(data.get("selected_preset") or self.current_preset_name).strip()
                if self.current_preset_name not in self.presets:
                    self.current_preset_name = "標準" if "標準" in self.presets else next(iter(self.presets))
            self.center_line_color = normalize_hex_color(data.get("center_line_color"), DEFAULT_CENTER_LINE_COLOR)
            geom = data.get("geometry")
            if isinstance(geom, dict):
                w = clamp_int(int(geom.get("width", self.width())), 640, 5000)
                h = clamp_int(int(geom.get("height", self.height())), 480, 4000)
                x = int(geom.get("x", self.x()))
                y = int(geom.get("y", self.y()))
                self.resize(w, h)
                self.move(self._safe_window_pos(x, y, w, h))
            sizes = data.get("splitter_sizes")
            if isinstance(sizes, list) and len(sizes) >= 2:
                self.splitter.setSizes([int(sizes[0]), int(sizes[1])])
        except Exception:
            self._settings_warning = "設定ファイルが壊れていたため初期設定で起動しました。"
        self._sync_param_controls()
        self.refresh_preset_ui(self.current_preset_name)
        self._update_center_line_color_widgets()
        if hasattr(self, "final_blur_enabled_check"):
            self.final_blur_enabled_check.blockSignals(True)
            self.final_blur_enabled_check.setChecked(bool(self.params.final_blur_enabled))
            self.final_blur_enabled_check.blockSignals(False)
        if hasattr(self, "center_line_enabled_check"):
            self.center_line_enabled_check.blockSignals(True)
            self.center_line_enabled_check.setChecked(bool(self.params.center_line_enabled))
            self.center_line_enabled_check.blockSignals(False)
        self._update_effect_group_enabled_states()
        self.canvas.set_brush_size(self.params.brush_size)

    def _safe_window_pos(self, x: int, y: int, w: int, h: int) -> QPoint:
        app = QApplication.instance()
        screen = app.primaryScreen() if app else None
        if screen is None:
            return QPoint(max(0, x), max(0, y))
        geo = screen.availableGeometry()
        x = clamp_int(x, geo.left(), max(geo.left(), geo.right() - min(w, geo.width()) + 1))
        y = clamp_int(y, geo.top(), max(geo.top(), geo.bottom() - min(h, geo.height()) + 1))
        return QPoint(x, y)

    def _save_settings(self) -> None:
        data = {
            "app": APP_NAME,
            "app_version": APP_VERSION,
            "params": asdict(self.params),
            "presets": self.presets,
            "selected_preset": self.current_preset_name,
            "center_line_color": self.center_line_color,
            "geometry": {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
            },
            "splitter_sizes": self.splitter.sizes(),
        }
        try:
            settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            self.statusBar().showMessage(f"設定保存に失敗しました: {exc}", 7000)

    def _sync_param_controls(self) -> None:
        for key, control in self.param_controls.items():
            control.set_value(getattr(self.params, key))

    def _update_action_states(self) -> None:
        has_image = self.original_rgba is not None
        self.save_action.setEnabled(has_image)
        self.save_as_action.setEnabled(has_image)
        self.undo_action.setEnabled(bool(self.undo_stack))
        self.redo_action.setEnabled(bool(self.redo_stack))

    def set_tool(self, tool: str) -> None:
        if tool not in {"brush", "eraser"}:
            tool = "brush"
        self.canvas.set_tool(tool)
        self.brush_btn.setChecked(tool == "brush")
        self.eraser_btn.setChecked(tool == "eraser")
        names = {"brush": "凹みブラシ", "eraser": "消しゴム"}
        self.statusBar().showMessage(f"ツール: {names.get(tool, tool)}", 3000)

    def on_final_blur_enabled_toggled(self, checked: bool) -> None:
        self.params.final_blur_enabled = 1 if checked else 0
        self._update_effect_group_enabled_states()
        self.request_render()
        self._save_settings()

    def on_center_line_enabled_toggled(self, checked: bool) -> None:
        self.params.center_line_enabled = 1 if checked else 0
        self._update_effect_group_enabled_states()
        self.request_render()
        self._save_settings()

    def _set_control_enabled(self, key: str, enabled: bool) -> None:
        control = self.param_controls.get(key)
        if control is not None:
            control.setEnabled(enabled)

    def _update_effect_group_enabled_states(self) -> None:
        blur_enabled = bool(self.params.final_blur_enabled)
        self._set_control_enabled("final_blur_size", blur_enabled)
        self._set_control_enabled("final_blur_strength", blur_enabled)
        line_enabled = bool(self.params.center_line_enabled)
        self._set_control_enabled("center_line_width", line_enabled)
        self._set_control_enabled("center_line_opacity", line_enabled)
        if hasattr(self, "center_line_color_edit"):
            self.center_line_color_edit.setEnabled(line_enabled)
        if hasattr(self, "center_line_color_btn"):
            self.center_line_color_btn.setEnabled(line_enabled)

    def _update_center_line_color_widgets(self) -> None:
        if hasattr(self, "center_line_color_edit"):
            self.center_line_color_edit.blockSignals(True)
            self.center_line_color_edit.setText(self.center_line_color)
            self.center_line_color_edit.blockSignals(False)
        if hasattr(self, "center_line_color_btn"):
            self.center_line_color_btn.setStyleSheet(
                f"background-color: {self.center_line_color}; border: 1px solid #666;"
            )

    def on_center_line_color_editing_finished(self) -> None:
        if not hasattr(self, "center_line_color_edit"):
            return
        new_color = normalize_hex_color(self.center_line_color_edit.text(), self.center_line_color)
        changed = new_color != self.center_line_color
        self.center_line_color = new_color
        self._update_center_line_color_widgets()
        if changed:
            self.schedule_render()
            self._save_settings()
            self.statusBar().showMessage(f"中心線の色を変更しました: {self.center_line_color}", 3000)

    def choose_center_line_color(self) -> None:
        current = QColor(self.center_line_color)
        color = QColorDialog.getColor(current, self, "中心線の色")
        if not color.isValid():
            return
        self.center_line_color = normalize_hex_color(color.name(QColor.NameFormat.HexRgb), self.center_line_color)
        self._update_center_line_color_widgets()
        self.schedule_render()
        self._save_settings()
        self.statusBar().showMessage(f"中心線の色を変更しました: {self.center_line_color}", 3000)

    def refresh_preset_ui(self, select_name: Optional[str] = None) -> None:
        if not hasattr(self, "preset_combo"):
            return
        if not self.presets:
            self.presets = sanitize_presets(PRESETS)
        name = select_name if select_name in self.presets else self.current_preset_name
        if name not in self.presets:
            name = "標準" if "標準" in self.presets else next(iter(self.presets))
        self.current_preset_name = name
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for preset_name in self.presets:
            self.preset_combo.addItem(preset_name)
        index = self.preset_combo.findText(name)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        line_edit = self.preset_combo.lineEdit()
        if line_edit is not None:
            line_edit.setText(name)
        self.preset_combo.blockSignals(False)
        self.del_preset_btn.setEnabled(len(self.presets) > 1)
        self.rebuild_preset_menu()

    def rebuild_preset_menu(self) -> None:
        if not hasattr(self, "preset_menu"):
            return
        self.preset_menu.clear()
        for name in self.presets:
            action = QAction(name, self)
            action.triggered.connect(lambda checked=False, n=name: self.apply_preset(n))
            self.preset_menu.addAction(action)

    def on_preset_activated(self, index: object) -> None:
        if isinstance(index, int):
            name = self.preset_combo.itemText(index).strip()
        else:
            name = str(index).strip()
        if name:
            self.apply_preset(name)

    def on_preset_name_editing_finished(self) -> None:
        line_edit = self.preset_combo.lineEdit()
        if line_edit is None:
            return
        typed = line_edit.text().strip()
        if not typed:
            self.refresh_preset_ui(self.current_preset_name)
            return
        if typed in self.presets:
            self.apply_preset(typed)
            return
        old_name = self.current_preset_name
        if old_name not in self.presets:
            self.presets[typed] = current_params_as_preset(self.params)
            self.current_preset_name = typed
        else:
            items = list(self.presets.items())
            unique_name = unique_preset_name(typed, [name for name in self.presets if name != old_name])
            self.presets = {
                (unique_name if name == old_name else name): values
                for name, values in items
            }
            self.current_preset_name = unique_name
        self.refresh_preset_ui(self.current_preset_name)
        self._save_settings()
        self.statusBar().showMessage(f"プリセット名を保存しました: {self.current_preset_name}", 3000)

    def new_preset_from_current(self) -> None:
        name = unique_preset_name("新規プリセット", self.presets)
        self.presets[name] = current_params_as_preset(self.params)
        self.current_preset_name = name
        self.refresh_preset_ui(name)
        self._save_settings()
        self.statusBar().showMessage(f"新しいプリセットを登録しました: {name}", 3000)

    def delete_current_preset(self) -> None:
        if len(self.presets) <= 1:
            self.statusBar().showMessage("最後のプリセットは削除できません。", 3000)
            return
        name = self.current_preset_name
        if name not in self.presets:
            self.refresh_preset_ui()
            return
        names = list(self.presets.keys())
        index = names.index(name)
        del self.presets[name]
        next_names = list(self.presets.keys())
        next_name = next_names[min(index, len(next_names) - 1)]
        self.current_preset_name = next_name
        self.refresh_preset_ui(next_name)
        self._save_settings()
        self.apply_preset(next_name)
        self.statusBar().showMessage(f"プリセットを削除しました: {name}", 3000)

    def apply_preset(self, name: str) -> None:
        preset = self.presets.get(name)
        if not preset:
            self.refresh_preset_ui(self.current_preset_name)
            return
        self.current_preset_name = name
        self.refresh_preset_ui(name)
        for key, value in preset.items():
            setattr(self.params, key, value)
        self.params.clamp_all()
        self._sync_param_controls()
        if hasattr(self, "final_blur_enabled_check"):
            self.final_blur_enabled_check.blockSignals(True)
            self.final_blur_enabled_check.setChecked(bool(self.params.final_blur_enabled))
            self.final_blur_enabled_check.blockSignals(False)
        if hasattr(self, "center_line_enabled_check"):
            self.center_line_enabled_check.blockSignals(True)
            self.center_line_enabled_check.setChecked(bool(self.params.center_line_enabled))
            self.center_line_enabled_check.blockSignals(False)
        self._update_effect_group_enabled_states()
        self.canvas.set_brush_size(self.params.brush_size)
        self._brush_stamp_cache_key = None
        self._brush_stamp_cache = None
        self.request_render()
        self._save_settings()
        self.statusBar().showMessage(f"プリセットを適用しました: {name}", 3000)

    def on_param_changed(self, key: str, value: int) -> None:
        setattr(self.params, key, value)
        self.params.clamp_all()
        if key == "brush_size":
            self.canvas.set_brush_size(self.params.brush_size)
        if key in {"brush_size", "brush_hardness", "brush_opacity"}:
            self._brush_stamp_cache_key = None
            self._brush_stamp_cache = None
        brush_only = {"brush_size", "brush_hardness", "brush_opacity", "brush_spacing", "smoothing"}
        if (
            self.original_rgba is not None
            and key not in {"jpeg_quality", "webp_quality"}
            and key not in brush_only
            and not self._stroke_active
        ):
            self.request_render()
        self._save_settings()

    def on_zoom_changed(self, zoom: float) -> None:
        self.update_info_label()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            return
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in SUPPORTED_EXTS:
                self.load_image_path(path)
                event.acceptProposedAction()
                return
        self.statusBar().showMessage("対応していないファイルです。", 5000)

    def open_image_dialog(self) -> None:
        start = str(self.image_path.parent if self.image_path else Path.home())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "画像を開く",
            start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )
        if path_str:
            self.load_image_path(Path(path_str))

    def open_project_dialog(self) -> None:
        start = str(self.project_path.parent if self.project_path else (self.image_path.parent if self.image_path else Path.home()))
        path_str, _ = QFileDialog.getOpenFileName(self, "プロジェクトを開く", start, PROJECT_FILE_FILTER)
        if path_str:
            self.load_project_path(Path(path_str))

    def save_project(self) -> None:
        if self.original_rgba is None:
            return
        target = self.project_path or self._default_project_path()
        self._save_project_to_path(target)

    def save_project_as(self) -> None:
        if self.original_rgba is None:
            return
        default = str(self.project_path or self._default_project_path())
        path_str, _ = QFileDialog.getSaveFileName(self, "プロジェクトに名前を付けて保存", default, PROJECT_FILE_FILTER)
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() != '.json' and not path.name.endswith('.dent.json'):
            path = path.with_suffix('.dent.json')
        self._save_project_to_path(path)

    def _save_project_to_path(self, path: Path) -> None:
        if self.original_rgba is None or self.mask is None:
            return
        try:
            project_data = {
                "app": APP_NAME,
                "app_version": APP_VERSION,
                "project_version": 1,
                "source_image_name": self.image_display_name,
                "source_image_path": str(self.image_path) if self.image_path else "",
                "image_size": {"width": int(self.original_rgba.shape[1]), "height": int(self.original_rgba.shape[0])},
                "image_png_base64": encode_rgba_png_base64(self.original_rgba),
                "params": asdict(self.params),
                "center_line_color": self.center_line_color,
                "selected_preset": self.current_preset_name,
                "strokes": [stroke.to_dict() for stroke in self.stroke_history],
            }
            path.write_text(json.dumps(project_data, ensure_ascii=False, indent=2), encoding='utf-8')
            self.project_path = path
            self.statusBar().showMessage(f"プロジェクトを保存しました: {path}", 8000)
            self.update_info_label()
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"プロジェクト保存に失敗しました: {exc}", 10000)

    def load_project_path(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('JSON形式が不正です。')
            image_b64 = data.get('image_png_base64')
            if not isinstance(image_b64, str) or not image_b64:
                raise ValueError('埋め込み画像がありません。')
            rgba = decode_rgba_png_base64(image_b64)
            source_name = str(data.get('source_image_name') or path.stem)
            source_path_raw = str(data.get('source_image_path') or '').strip()
            source_path = Path(source_path_raw) if source_path_raw else None
            if source_path is not None and not source_path.exists():
                source_path = None
            if isinstance(data.get('params'), dict):
                self.params = ToolParams.from_dict(data.get('params'))
            self.center_line_color = normalize_hex_color(data.get('center_line_color'), DEFAULT_CENTER_LINE_COLOR)
            selected_preset = str(data.get('selected_preset') or self.current_preset_name).strip()
            if selected_preset in self.presets:
                self.current_preset_name = selected_preset
            self.load_image_array(rgba, source_path, source_name, keep_project_path=True)
            raw_strokes = data.get('strokes', [])
            loaded_strokes: List[StrokeRecord] = []
            if isinstance(raw_strokes, list):
                for item in raw_strokes:
                    stroke = StrokeRecord.from_dict(item)
                    if stroke is not None:
                        loaded_strokes.append(stroke)
            self.stroke_history = loaded_strokes
            self.mask = replay_mask_from_strokes(self.original_rgba.shape[:2], self.stroke_history)
            self.project_path = path
            self._sync_param_controls()
            if hasattr(self, "final_blur_enabled_check"):
                self.final_blur_enabled_check.blockSignals(True)
                self.final_blur_enabled_check.setChecked(bool(self.params.final_blur_enabled))
                self.final_blur_enabled_check.blockSignals(False)
            if hasattr(self, "center_line_enabled_check"):
                self.center_line_enabled_check.blockSignals(True)
                self.center_line_enabled_check.setChecked(bool(self.params.center_line_enabled))
                self.center_line_enabled_check.blockSignals(False)
            self._update_center_line_color_widgets()
            self._update_effect_group_enabled_states()
            self.request_render()
            self.statusBar().showMessage(f"プロジェクトを読み込みました: {path}", 8000)
            self.update_info_label()
            self._update_action_states()
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"プロジェクト読み込みに失敗しました: {exc}", 10000)

    def paste_from_clipboard(self) -> None:
        cb = QApplication.clipboard()
        mime = cb.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if path.suffix.lower() in SUPPORTED_EXTS:
                        self.load_image_path(path)
                        return
        qimg = cb.image()
        if not qimg.isNull():
            rgba = qimage_to_rgba(qimg)
            self.load_image_array(rgba, None, "クリップボード画像")
            return
        self.statusBar().showMessage("クリップボードに画像がありません。", 5000)

    def load_image_path(self, path: Path) -> None:
        try:
            rgba = cv2_read_image_unicode(path)
        except Exception as exc:
            self.statusBar().showMessage(f"画像の読み込みに失敗しました: {exc}", 8000)
            return
        self.load_image_array(rgba, path, str(path))

    def load_image_array(self, rgba: np.ndarray, path: Optional[Path], label: str, keep_project_path: bool = False) -> None:
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            self.statusBar().showMessage("RGBA画像への変換に失敗しました。", 7000)
            return
        self.original_rgba = np.ascontiguousarray(rgba.astype(np.uint8))
        self.preview_rgba = self.original_rgba.copy()
        self.mask = np.zeros(self.original_rgba.shape[:2], dtype=np.uint8)
        self.image_path = path
        self.image_display_name = Path(label).name if label else (path.name if path else "クリップボード画像")
        if not keep_project_path:
            self.project_path = None
        self.stroke_history = []
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._last_paint = None
        self._smoothed_paint = None
        self.canvas.set_image(ndarray_rgba_to_qimage(self.preview_rgba))
        self.statusBar().showMessage(f"画像を読み込みました: {label}", 5000)
        self.update_info_label()
        self._update_action_states()

    def update_info_label(self) -> None:
        if self.original_rgba is None:
            self.info_label.setText("未読み込み")
            return
        h, w = self.original_rgba.shape[:2]
        name = self.image_display_name if self.image_display_name else (self.image_path.name if self.image_path else "クリップボード画像")
        self.info_label.setText(
            f"{name}\n{w} x {h}px\nズーム {self.canvas.zoom * 100:.1f}%\n"
            f"Undo {len(self.undo_stack)} / Redo {len(self.redo_stack)}"
        )

    def current_brush_stamp(self) -> np.ndarray:
        key = (self.params.brush_size, self.params.brush_hardness, self.params.brush_opacity)
        if self._brush_stamp_cache_key != key or self._brush_stamp_cache is None:
            self._brush_stamp_cache = create_brush_stamp(
                self.params.brush_size,
                self.params.brush_hardness,
                self.params.brush_opacity,
            )
            self._brush_stamp_cache_key = key
        return self._brush_stamp_cache

    def _make_state_snapshot(self) -> Dict[str, object]:
        return {
            "mask": self.mask.copy() if self.mask is not None else None,
            "strokes": clone_stroke_records(self.stroke_history),
        }

    def _restore_state_snapshot(self, snapshot: Dict[str, object]) -> None:
        mask = snapshot.get("mask")
        self.mask = mask.copy() if isinstance(mask, np.ndarray) else None
        raw_strokes = snapshot.get("strokes", [])
        self.stroke_history = clone_stroke_records(raw_strokes) if isinstance(raw_strokes, list) else []

    def _default_project_path(self) -> Path:
        base_dir = self.image_path.parent if self.image_path else (self.project_path.parent if self.project_path else app_dir())
        stem = self.image_path.stem if self.image_path else (self.project_path.stem.replace('.dent', '') if self.project_path else 'dent-project')
        candidate = base_dir / f"{stem}.dent.json"
        if not candidate.exists() or candidate == self.project_path:
            return candidate
        for i in range(2, 10000):
            candidate = base_dir / f"{stem}_v{i}.dent.json"
            if not candidate.exists():
                return candidate
        raise RuntimeError("プロジェクトファイル名の作成に失敗しました。")

    def on_stroke_started(self) -> None:
        if self.mask is None:
            return
        self._stroke_active = True
        self._stroke_before_state = self._make_state_snapshot()
        self._current_stroke_points = []
        self._current_stroke_erase = (self.canvas.tool == "eraser")
        self._last_paint = None
        self._smoothed_paint = None
        self.canvas.begin_live_stroke(self.canvas.tool == "eraser")

    def on_stroke_painted(self, x: float, y: float, erase: bool) -> None:
        if self.mask is None:
            return
        if self.original_rgba is None:
            return
        x = max(0.0, min(float(self.mask.shape[1] - 1), x))
        y = max(0.0, min(float(self.mask.shape[0] - 1), y))
        if self.params.smoothing > 0 and self._smoothed_paint is not None:
            a = 1.0 - (self.params.smoothing / 100.0) * 0.75
            sx = self._smoothed_paint[0] * (1.0 - a) + x * a
            sy = self._smoothed_paint[1] * (1.0 - a) + y * a
            x, y = sx, sy
        self._smoothed_paint = (x, y)

        stamp = self.current_brush_stamp()
        spacing_px = max(1.0, self.params.brush_size * self.params.brush_spacing / 100.0)
        if self._last_paint is None:
            blend_stamp(self.mask, stamp, x, y, erase)
            self._last_paint = (x, y)
        else:
            lx, ly = self._last_paint
            dx = x - lx
            dy = y - ly
            dist = math.hypot(dx, dy)
            steps = max(1, int(dist / spacing_px))
            for i in range(1, steps + 1):
                t = i / steps
                px = lx + dx * t
                py = ly + dy * t
                blend_stamp(self.mask, stamp, px, py, erase)
            self._last_paint = (x, y)
        self._current_stroke_points.append((x, y))
        self.canvas.add_live_stroke_point(x, y, erase)

    def on_stroke_finished(self) -> None:
        if self.mask is None:
            return
        if self._stroke_before_state is not None:
            before_mask = self._stroke_before_state.get("mask")
            if isinstance(before_mask, np.ndarray) and not np.array_equal(before_mask, self.mask):
                self.undo_stack.append(self._stroke_before_state)
                if len(self.undo_stack) > self.max_undo:
                    self.undo_stack.pop(0)
                self.redo_stack.clear()
                if self._current_stroke_points:
                    self.stroke_history.append(
                        StrokeRecord(
                            erase=self._current_stroke_erase,
                            points=list(self._current_stroke_points),
                            brush_size=self.params.brush_size,
                            brush_hardness=self.params.brush_hardness,
                            brush_opacity=self.params.brush_opacity,
                            brush_spacing=self.params.brush_spacing,
                        )
                    )
        self._stroke_before_state = None
        self._current_stroke_points = []
        self._stroke_active = False
        self._last_paint = None
        self._smoothed_paint = None
        self.request_render(light=False)
        self.canvas.end_live_stroke()
        self._update_action_states()
        self.update_info_label()

    def request_render(self, light: bool = False) -> None:
        if self.original_rgba is None or self.mask is None:
            return
        if light:
            if not self._render_pending:
                self._render_pending = True
                QTimer.singleShot(25, self.render_preview)
        else:
            self.render_preview()

    def render_preview(self) -> None:
        self._render_pending = False
        if self.original_rgba is None or self.mask is None:
            return
        try:
            self.preview_rgba = apply_dent_effect(self.original_rgba, self.mask, self.params, self.center_line_color, self.stroke_history)
            if self._show_before:
                self.canvas.replace_image_without_fit(ndarray_rgba_to_qimage(self.original_rgba))
            else:
                self.canvas.replace_image_without_fit(ndarray_rgba_to_qimage(self.preview_rgba))
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"プレビュー生成に失敗しました: {exc}", 8000)

    def toggle_before_after(self) -> None:
        if self.original_rgba is None:
            return
        self._show_before = not self._show_before
        if self._show_before:
            self.canvas.replace_image_without_fit(ndarray_rgba_to_qimage(self.original_rgba))
            self.statusBar().showMessage("Before表示", 2000)
        else:
            if self.preview_rgba is None:
                self.render_preview()
            else:
                self.canvas.replace_image_without_fit(ndarray_rgba_to_qimage(self.preview_rgba))
            self.statusBar().showMessage("After表示", 2000)

    def undo(self) -> None:
        if self.mask is None or not self.undo_stack:
            return
        self.redo_stack.append(self._make_state_snapshot())
        snapshot = self.undo_stack.pop()
        self._restore_state_snapshot(snapshot)
        self.request_render()
        self._update_action_states()
        self.update_info_label()
        self.statusBar().showMessage("元に戻しました。", 2000)

    def redo(self) -> None:
        if self.mask is None or not self.redo_stack:
            return
        self.undo_stack.append(self._make_state_snapshot())
        snapshot = self.redo_stack.pop()
        self._restore_state_snapshot(snapshot)
        self.request_render()
        self._update_action_states()
        self.update_info_label()
        self.statusBar().showMessage("やり直しました。", 2000)

    def clear_all_mask_confirm(self) -> None:
        if self.mask is None or np.max(self.mask) == 0:
            return
        ret = QMessageBox.question(
            self,
            "全ての凹みをクリア",
            "現在の凹み編集を全てクリアします。よろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.undo_stack.append(self._make_state_snapshot())
        self.redo_stack.clear()
        self.mask[:] = 0
        self.stroke_history = []
        self.request_render()
        self._update_action_states()
        self.update_info_label()
        self.statusBar().showMessage("全ての凹みをクリアしました。", 3000)

    def save_image(self) -> None:
        if self.original_rgba is None:
            return
        default = unique_output_path(self.image_path, ".png")
        self._save_to_path(default)

    def save_image_as(self) -> None:
        if self.original_rgba is None:
            return
        default = str(unique_output_path(self.image_path, ".png"))
        path_str, selected_filter = QFileDialog.getSaveFileName(
            self,
            "名前を付けて保存",
            default,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)",
        )
        if not path_str:
            return
        path = Path(path_str)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            if "JPEG" in selected_filter:
                path = path.with_suffix(".jpg")
            elif "WEBP" in selected_filter:
                path = path.with_suffix(".webp")
            else:
                path = path.with_suffix(".png")
        self._save_to_path(path)

    def _save_to_path(self, path: Path) -> None:
        if self.original_rgba is None:
            return
        try:
            if self.preview_rgba is None:
                self.preview_rgba = apply_dent_effect(self.original_rgba, self.mask, self.params, self.center_line_color, self.stroke_history) if self.mask is not None else self.original_rgba
            suffix = path.suffix.lower()
            quality = self.params.jpeg_quality if suffix in {".jpg", ".jpeg"} else self.params.webp_quality
            cv2_write_image_unicode(path, self.preview_rgba, quality=quality)
            if not path.exists():
                raise FileNotFoundError("保存後のファイルが存在しません。")
            check = cv2_read_image_unicode(path)
            if check.size == 0:
                raise ValueError("保存後の再読み込み確認に失敗しました。")
            if suffix == ".png":
                pil = Image.open(path)
                has_alpha = pil.mode in {"RGBA", "LA"} or (pil.mode == "P" and "transparency" in pil.info)
                if self.preview_rgba.shape[2] == 4 and np.any(self.preview_rgba[:, :, 3] < 255) and not has_alpha:
                    raise ValueError("透過PNGのalpha確認に失敗しました。")
            self.statusBar().showMessage(f"保存しました: {path}", 8000)
        except Exception as exc:
            traceback.print_exc()
            self.statusBar().showMessage(f"保存に失敗しました: {exc}", 10000)

    def show_canvas_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        action_before = menu.addAction("元画像を表示 / 戻す")
        action_fit = menu.addAction("全体表示")
        action_100 = menu.addAction("100%表示")
        menu.addSeparator()
        action_undo = menu.addAction("現在のストロークを取り消し")
        action_clear = menu.addAction("全ての凹みをクリア")
        menu.addSeparator()
        action_save = menu.addAction("保存")
        action_save_as = menu.addAction("名前を付けて保存")
        action_undo.setEnabled(bool(self.undo_stack))
        action_clear.setEnabled(self.mask is not None and np.max(self.mask) > 0)
        action_save.setEnabled(self.original_rgba is not None)
        action_save_as.setEnabled(self.original_rgba is not None)
        chosen = menu.exec(global_pos)
        if chosen == action_before:
            self.toggle_before_after()
        elif chosen == action_fit:
            self.canvas.fit_to_window()
        elif chosen == action_100:
            self.canvas.set_actual_size()
        elif chosen == action_undo:
            self.undo()
        elif chosen == action_clear:
            self.clear_all_mask_confirm()
        elif chosen == action_save:
            self.save_image()
        elif chosen == action_save_as:
            self.save_image_as()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        # QActions handle Ctrl+Z/Ctrl+S/etc. Plain tool shortcuts are handled here
        # only when text widgets do not own focus.
        focus = QApplication.focusWidget()
        if focus and focus.metaObject().className() in {"QLineEdit", "QTextEdit", "QPlainTextEdit", "QSpinBox"}:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_1:
            self.set_tool("brush")
            return
        if event.key() == Qt.Key.Key_2:
            self.set_tool("eraser")
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_settings()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
