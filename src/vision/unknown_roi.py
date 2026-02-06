from __future__ import annotations

"""ROI 内未知物体回退检测。

设计目标：
1. 当 ROI 内没有任何已知类别检测结果时，尝试用轮廓法补一个 unknown；
2. 只输出带 mask 的 unknown，便于沿用现有接触判定（mask 优先）与可视化流程；
3. 使用手关键点凸包作为排除区，尽量避免把“手本身”误标成 unknown。
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..types import DetectedObject, HandKeypoints


@dataclass(frozen=True)
class UnknownRoiOptions:
    """ROI 未知物体回退检测参数。"""

    enabled: bool = True
    min_area_ratio: float = 0.012
    max_area_ratio: float = 0.55
    max_hand_dist_ratio: float = 0.58
    hand_exclude_dilate: int = 7
    max_hand_overlap_ratio: float = 0.68
    min_fill_ratio: float = 0.22
    min_solidity: float = 0.55
    max_aspect_ratio: float = 3.2
    border_margin: int = 3
    min_score: float = 0.12
    max_score: float = 0.30


def infer_unknown_object_on_roi(
    frame_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    crop_bgr: np.ndarray,
    hands_lr: dict[str, HandKeypoints | None],
    options: UnknownRoiOptions,
) -> DetectedObject | None:
    """在单个 ROI 内推断 unknown 目标。

    仅在“已知类别没有检测到结果”时调用该函数。

    参数:
        frame_shape: 原图尺寸 (H, W)。
        roi: ROI 坐标 (x1, y1, x2, y2)。
        crop_bgr: ROI 对应的图像裁剪（BGR）。
        hands_lr: 左右手关键点映射，用于构建手部排除区。
        options: 未知检测参数。

    返回:
        DetectedObject | None: unknown 目标；若未找到稳定候选则返回 None。
    """
    if (not options.enabled) or crop_bgr.size == 0:
        return None

    x1, y1, x2, y2 = roi
    roi_h = max(0, y2 - y1)
    roi_w = max(0, x2 - x1)
    if roi_h < 8 or roi_w < 8:
        return None

    safe = _sanitize_options(options)
    hand_mask, hand_center = _build_hand_exclude_mask(
        roi=roi,
        roi_shape=(roi_h, roi_w),
        hands_lr=hands_lr,
        dilate_px=safe.hand_exclude_dilate,
    )
    # 若无法可靠定位手中心，则不触发 unknown 回退，避免静态背景误检。
    if hand_center is None:
        return None

    binary = _build_edge_binary(crop_bgr, hand_mask=hand_mask)
    candidate = _pick_best_candidate(
        binary=binary,
        hand_mask=hand_mask,
        hand_center=hand_center,
        min_area_ratio=safe.min_area_ratio,
        max_area_ratio=safe.max_area_ratio,
        max_hand_dist_ratio=safe.max_hand_dist_ratio,
        max_hand_overlap_ratio=safe.max_hand_overlap_ratio,
        min_fill_ratio=safe.min_fill_ratio,
        min_solidity=safe.min_solidity,
        max_aspect_ratio=safe.max_aspect_ratio,
        border_margin=safe.border_margin,
        min_score=safe.min_score,
        max_score=safe.max_score,
    )
    if candidate is None:
        return None

    local_bbox, local_mask, score = candidate
    full_h, full_w = frame_shape
    mask_full = np.zeros((full_h, full_w), dtype=np.float32)

    rx2 = min(x1 + roi_w, full_w)
    ry2 = min(y1 + roi_h, full_h)
    if rx2 <= x1 or ry2 <= y1:
        return None

    mask_full[y1:ry2, x1:rx2] = local_mask[: ry2 - y1, : rx2 - x1]
    lx1, ly1, lx2, ly2 = local_bbox
    mapped_bbox = (
        float(lx1 + x1),
        float(ly1 + y1),
        float(lx2 + x1),
        float(ly2 + y1),
    )

    return DetectedObject(
        name="unknown",
        bbox=mapped_bbox,
        score=float(score),
        mask=mask_full,
    )


def _sanitize_options(options: UnknownRoiOptions) -> UnknownRoiOptions:
    """对参数做边界保护，防止异常输入导致不稳定行为。"""
    min_area_ratio = float(max(0.001, min(options.min_area_ratio, 0.9)))
    max_area_ratio = float(max(min_area_ratio + 0.01, min(options.max_area_ratio, 0.98)))
    max_hand_dist_ratio = float(max(0.1, min(options.max_hand_dist_ratio, 2.0)))
    hand_exclude_dilate = int(max(0, min(options.hand_exclude_dilate, 64)))
    max_hand_overlap_ratio = float(max(0.05, min(options.max_hand_overlap_ratio, 0.98)))
    min_fill_ratio = float(max(0.01, min(options.min_fill_ratio, 0.99)))
    min_solidity = float(max(0.05, min(options.min_solidity, 0.99)))
    max_aspect_ratio = float(max(1.2, min(options.max_aspect_ratio, 12.0)))
    border_margin = int(max(0, min(options.border_margin, 24)))
    min_score = float(max(0.01, min(options.min_score, 0.99)))
    max_score = float(max(min_score + 0.01, min(options.max_score, 0.99)))
    return UnknownRoiOptions(
        enabled=bool(options.enabled),
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_hand_dist_ratio=max_hand_dist_ratio,
        hand_exclude_dilate=hand_exclude_dilate,
        max_hand_overlap_ratio=max_hand_overlap_ratio,
        min_fill_ratio=min_fill_ratio,
        min_solidity=min_solidity,
        max_aspect_ratio=max_aspect_ratio,
        border_margin=border_margin,
        min_score=min_score,
        max_score=max_score,
    )


def _build_hand_exclude_mask(
    roi: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
    hands_lr: dict[str, HandKeypoints | None],
    dilate_px: int,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """构建 ROI 内手部排除区（凸包 + 膨胀）与手中心。

    说明：
    - 排除区用于减少“把手误检成 unknown”；
    - 手中心用于约束 unknown 候选距离，抑制远离手部的噪声轮廓。
    """
    x1, y1, _, _ = roi
    roi_h, roi_w = roi_shape
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    centers: list[tuple[float, float]] = []

    for side in ("left", "right"):
        kps = hands_lr.get(side)
        if not kps:
            continue

        pts_local: list[tuple[float, float]] = []
        for x, y, conf in kps:
            if conf <= 0:
                continue
            lx = float(x - x1)
            ly = float(y - y1)

            # 允许少量越界点参与凸包，增强边缘场景鲁棒性。
            if -12.0 <= lx <= (roi_w + 12.0) and -12.0 <= ly <= (roi_h + 12.0):
                pts_local.append((lx, ly))

            if 0.0 <= lx < roi_w and 0.0 <= ly < roi_h:
                centers.append((lx, ly))

        if len(pts_local) >= 3:
            hull = cv2.convexHull(np.array(pts_local, dtype=np.float32))
            hull_i = np.round(hull).astype(np.int32)
            cv2.fillConvexPoly(mask, hull_i, 255)

    if np.count_nonzero(mask) <= 0:
        center = _mean_point(centers)
        return None, center

    if dilate_px > 0:
        k = int(max(1, dilate_px * 2 + 1))
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    center = _mean_point(centers)
    return mask, center


def _build_edge_binary(crop_bgr: np.ndarray, hand_mask: np.ndarray | None) -> np.ndarray:
    """构建候选二值边缘图。"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median_val = float(np.median(blur))
    low = int(max(18, 0.66 * median_val))
    high = int(min(255, max(low + 18, 1.33 * median_val)))

    edges = cv2.Canny(blur, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    return edges


def _pick_best_candidate(
    binary: np.ndarray,
    hand_mask: np.ndarray | None,
    hand_center: tuple[float, float] | None,
    min_area_ratio: float,
    max_area_ratio: float,
    max_hand_dist_ratio: float,
    max_hand_overlap_ratio: float,
    min_fill_ratio: float,
    min_solidity: float,
    max_aspect_ratio: float,
    border_margin: int,
    min_score: float,
    max_score: float,
) -> tuple[tuple[int, int, int, int], np.ndarray, float] | None:
    """从轮廓中选择 unknown 候选。"""
    roi_h, roi_w = binary.shape[:2]
    roi_area = float(max(1, roi_h * roi_w))
    min_area = roi_area * min_area_ratio
    max_area = roi_area * max_area_ratio
    roi_diag = float((roi_w * roi_w + roi_h * roi_h) ** 0.5)
    max_hand_dist = max(1e-6, max_hand_dist_ratio * roi_diag)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: tuple[tuple[int, int, int, int], np.ndarray, float] | None = None
    best_score = -1.0

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area or contour_area > max_area:
            continue

        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        pix = int(np.count_nonzero(mask))
        if pix < min_area or pix > max_area:
            continue

        if hand_mask is not None:
            hand_overlap = float(np.count_nonzero((mask > 0) & (hand_mask > 0)) / max(1, pix))
            if hand_overlap > max_hand_overlap_ratio:
                continue

        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue

        bx1 = int(xs.min())
        by1 = int(ys.min())
        bx2 = int(xs.max()) + 1
        by2 = int(ys.max()) + 1
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 8 or bh < 8:
            continue
        if (
            bx1 <= border_margin
            or by1 <= border_margin
            or bx2 >= (roi_w - border_margin)
            or by2 >= (roi_h - border_margin)
        ):
            # 紧贴 ROI 边界的轮廓常见于墙缝/衣物边缘等背景结构，优先剔除。
            continue

        bbox_area = float(max(1, bw * bh))
        fill_ratio = float(pix / bbox_area)
        if fill_ratio < min_fill_ratio:
            continue

        aspect_ratio = float(max(bw / max(1.0, bh), bh / max(1.0, bw)))
        if aspect_ratio > max_aspect_ratio:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(max(1e-6, cv2.contourArea(hull)))
        solidity = float(min(1.0, pix / hull_area))
        if solidity < min_solidity:
            continue

        cx = float(xs.mean())
        cy = float(ys.mean())
        if hand_center is not None:
            dist = float(((cx - hand_center[0]) ** 2 + (cy - hand_center[1]) ** 2) ** 0.5)
            if dist > max_hand_dist:
                continue
            dist_score = 1.0 - min(1.0, dist / max_hand_dist)
        else:
            dist_score = 0.5

        perimeter = float(max(1e-6, cv2.arcLength(contour, True)))
        compactness = float(max(0.0, min(1.0, 4.0 * np.pi * pix / (perimeter * perimeter))))
        area_score = float(min(1.0, pix / max(1e-6, roi_area * 0.35)))
        shape_score = 0.55 * compactness + 0.45 * min(1.0, solidity)

        quality = 0.40 * area_score + 0.35 * shape_score + 0.25 * dist_score
        score = float(min_score + (max_score - min_score) * max(0.0, min(1.0, quality)))

        if score <= best_score:
            continue

        best_score = score
        best = ((bx1, by1, bx2, by2), (mask > 0).astype(np.float32), score)

    return best


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """计算点集均值中心。"""
    if not points:
        return None
    return (
        float(sum(p[0] for p in points) / len(points)),
        float(sum(p[1] for p in points) / len(points)),
    )
