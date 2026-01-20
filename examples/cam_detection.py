import glob
import os
import platform
import stat
import threading
import time

import cv2


BACKENDS = [
    ("V4L2", cv2.CAP_V4L2),
    ("ANY", cv2.CAP_ANY),
]


def _read_frame(cap, result):
    ok, frame = cap.read()
    result["ok"] = ok
    result["frame"] = frame


def _set_timeout(cap, timeout_ms):
    for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        if hasattr(cv2, prop):
            cap.set(getattr(cv2, prop), timeout_ms)


def _backend_name(cap):
    backend_id = int(cap.get(cv2.CAP_PROP_BACKEND))
    for name, value in BACKENDS:
        if backend_id == value:
            return name
    return str(backend_id)


def probe_source(source, api_preference, timeout_sec=2.0):
    cap = cv2.VideoCapture(source, api_preference)
    if not cap.isOpened():
        cap.release()
        return "NO_OPEN", ""

    _set_timeout(cap, int(timeout_sec * 1000))

    result = {"ok": False, "frame": None}
    reader = threading.Thread(target=_read_frame, args=(cap, result), daemon=True)
    reader.start()
    reader.join(timeout_sec)

    backend = _backend_name(cap)
    cap.release()

    if reader.is_alive():
        return "TIMEOUT", backend
    if result["ok"] and result["frame"] is not None:
        height, width = result["frame"].shape[:2]
        return "OK", f"{backend} {width}x{height}"
    return "NO_READ", backend


def print_device_nodes():
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* devices found.")
        return devices
    print("Device nodes:")
    for device in devices:
        try:
            st = os.stat(device)
            perms = stat.filemode(st.st_mode)
            print(f"  {device} {perms} gid={st.st_gid}")
        except OSError as exc:
            print(f"  {device} <stat failed: {exc}>")
    return devices


def print_environment_info():
    print("Environment:")
    print(f"  Python: {platform.python_version()}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  Platform: {platform.platform()}")


def main():
    print_environment_info()
    devices = print_device_nodes()

    print("\nProbe by device path:")
    if not devices:
        print("  (skip: no /dev/video* nodes)")
    for device in devices:
        for name, backend in BACKENDS:
            status, info = probe_source(device, backend)
            print(f"  {device} [{name}] {status} {info}")
        time.sleep(0.1)

    print("\nProbe by index:")
    max_index = max(5, len(devices) + 1)
    for index in range(max_index):
        for name, backend in BACKENDS:
            status, info = probe_source(index, backend)
            print(f"  {index} [{name}] {status} {info}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
