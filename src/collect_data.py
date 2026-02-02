#!/usr/bin/env python3
"""Data collection script for capturing hand sign images from webcam."""

import time
from pathlib import Path
import cv2

from src.utils.config import load_config, create_directories


def save_frame(img, out_dir: Path, sign: str) -> Path:
    """Save a single frame to disk and return the path."""
    ts = int(time.time() * 1000)
    fname = out_dir / f"{sign}_{ts}.jpg"
    cv2.imwrite(str(fname), img)
    return fname


def main():
    config = load_config()
    create_directories(config)

    raw_root = Path(config["paths"]["raw_data"])
    signs = config["collection"]["signs"]

    # Ensure sign-specific subdirectories exist
    for s in signs:
        (raw_root / s).mkdir(parents=True, exist_ok=True)

    image_width = int(config["collection"]["image_width"])
    image_height = int(config["collection"]["image_height"])
    capture_interval = float(config["collection"]["capture_interval"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    last_saved = 0.0
    print(f"Press the sign key ({', '.join(signs)}) to save a frame to {raw_root}/<sign>. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # If a specific sign key is pressed, save the frame (respecting capture interval)
        if key != 255:
            try:
                kchr = chr(key).upper()
            except Exception:
                kchr = ""

            if kchr in [s.upper() for s in signs]:
                now = time.time()
                if now - last_saved >= capture_interval:
                    out_dir = raw_root / kchr
                    out_dir.mkdir(parents=True, exist_ok=True)
                    saved = save_frame(img, out_dir, kchr)
                    print(f"Saved {saved}")
                    last_saved = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
