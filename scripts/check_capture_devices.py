#!/usr/bin/env python3
"""Inspect WSL capture/audio devices; --strict actively probes startup readiness."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversation import (
    _quiet_native_audio_stderr,
    available_microphones,
    camera_candidates,
    choose_microphone_index,
    cv2,
    sr,
)  # noqa: E402


def probe_camera(cameras: list[tuple[str, str]]) -> tuple[bool, str]:
    if cv2 is None:
        return False, "OpenCV camera capture is not installed"
    last_error = ""
    for path, name in cameras:
        capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
        try:
            if not capture.isOpened():
                continue
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            frame = None
            for _ in range(4):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    frame = candidate
                time.sleep(0.05)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            if width >= 640 and height >= 360:
                return True, f"{path} · {name} · {width}×{height} frame received"
        except Exception as error:
            last_error = str(error)
        finally:
            capture.release()
    detail = f" ({last_error})" if last_error else ""
    return False, f"no camera returned a usable 640×360 frame{detail}"


def probe_microphone(selected: int | None) -> tuple[bool, str]:
    if sr is None:
        return False, "SpeechRecognition/PyAudio is not installed"
    if selected is None:
        return False, "no PyAudio input device detected"
    try:
        with _quiet_native_audio_stderr():
            with sr.Microphone(device_index=selected, sample_rate=16000) as source:
                if source.stream is None:
                    return False, "selected microphone did not open a capture stream"
        return True, "selected PyAudio microphone stream opened successfully"
    except Exception as error:
        return False, f"selected microphone could not be opened: {error}"


def probe_laptop_output() -> tuple[bool, str]:
    if not shutil.which("paplay"):
        return False, "paplay is not installed"
    pactl = shutil.which("pactl")
    if not pactl:
        return True, "paplay is installed; using the default PulseAudio sink"
    try:
        result = subprocess.run(
            [pactl, "info"], capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"PulseAudio connection check failed: {error}"
    if result.returncode != 0:
        return False, "could not connect to the WSLg PulseAudio server"
    sink = next(
        (line.partition(":")[2].strip() for line in result.stdout.splitlines()
         if line.startswith("Default Sink:")),
        "PulseAudio default sink",
    )
    return True, f"{sink}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="actively open camera, microphone, and laptop audio; exit nonzero if any are unavailable",
    )
    args = parser.parse_args()

    cameras = camera_candidates()
    print(f"OpenCV camera capture: {cv2.__version__ if cv2 is not None else 'not installed'}")
    print("WSL cameras:")
    if cameras:
        for path, name in cameras:
            print(f"  {path} · {name}")
    else:
        print("  none found (attach a camera with usbipd first)")

    print("\nWSL microphone inputs:")
    devices: list[dict] = []
    default_index = None
    if sr is None:
        print("  SpeechRecognition/PyAudio is not installed")
    else:
        devices, default_index = available_microphones()
        for item in devices:
            index = item["index"]
            tag = " (default)" if index == default_index else ""
            print(f"  [{index}] {item.get('name', 'unnamed input')}{tag}")
        selected = choose_microphone_index(devices, default_index)
        if selected is None:
            print("  no PyAudio input device detected")
        else:
            selected_name = next(item.get("name", "input") for item in devices if item["index"] == selected)
            print(f"  Headroom selection: [{selected}] {selected_name}")

    pulse = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, check=False) if shutil.which("pactl") else None
    if pulse and pulse.returncode == 0:
        names = [line.split()[1] for line in pulse.stdout.splitlines() if len(line.split()) >= 2 and ".monitor" not in line.split()[1]]
        print("\nPulseAudio sources:")
        if names:
            for name in names:
                print(f"  {name}")
        else:
            print("  no microphone source found")
    else:
        print("\nPulseAudio source listing unavailable; Headroom still checks PyAudio inputs.")

    if not args.strict:
        return 0

    print("\nActive bridge checks:")
    camera_ok, camera_message = probe_camera(cameras)
    mic_ok, mic_message = probe_microphone(choose_microphone_index(devices, default_index))
    output_ok, output_message = probe_laptop_output()
    print(f"  Camera: {'READY' if camera_ok else 'NOT READY'} · {camera_message}")
    print(f"  Microphone: {'READY' if mic_ok else 'NOT READY'} · {mic_message}")
    print(f"  Laptop output: {'READY' if output_ok else 'NOT READY'} · {output_message}")
    if camera_ok and mic_ok and output_ok:
        print("Hardware preflight: PASS")
        return 0
    print("Hardware preflight: WAITING (one or more bridges are not ready)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
