from __future__ import annotations

"""摄像头输入封装。"""

import cv2

from ..config import is_windows


def backend_candidates() -> list[tuple[str, int]]:
    """获取当前平台推荐的视频后端列表（按优先级排序）。

    返回:
        list[tuple[str, int]]: (名称, OpenCV 后端常量) 列表。
    """
    if is_windows():
        candidates: list[tuple[str, int]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("MSMF", cv2.CAP_MSMF))
        candidates.append(("ANY", cv2.CAP_ANY))
        return candidates

    candidates = []
    if hasattr(cv2, "CAP_V4L2"):
        candidates.append(("V4L2", cv2.CAP_V4L2))
    candidates.append(("ANY", cv2.CAP_ANY))
    return candidates


def open_capture(source: int | str) -> tuple[cv2.VideoCapture | None, str]:
    """打开摄像头或视频流。

    参数:
        source: 摄像头索引（int）或设备路径/URL（str）。

    返回:
        tuple[cv2.VideoCapture | None, str]: (cap, backend_name)。
    """
    last_backend = "ANY"
    for name, backend in backend_candidates():
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap, name
        cap.release()
        last_backend = name
    return None, last_backend
