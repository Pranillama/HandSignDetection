#!/usr/bin/env python3
"""Data collection script for capturing hand sign images from webcam.

Supports burst capture mode via CLI argument for efficient data collection.
Includes on-screen overlay with instructions and collection statistics.
"""

import argparse
import time
from pathlib import Path
from datetime import datetime
import cv2
import logging

from src.utils.config import load_config, create_directories
from src.utils.logger import setup_logger


def save_frame(img, out_dir: Path, sign: str, index: int) -> Path:
    """Save a single frame to disk with timestamp and index.
    
    Args:
        img: OpenCV image frame to save.
        out_dir: Directory where the image will be saved.
        sign: Hand sign label.
        index: Capture index for this burst.
        
    Returns:
        Path to the saved image file.
    """
    ts = int(time.time() * 1000)
    fname = out_dir / f"{sign}_{ts}_{index}.jpg"
    cv2.imwrite(str(fname), img)
    return fname


def draw_overlay(img, sign: str, burst_mode: bool, frame_count: int, total_frames: int,
                 start_time: float, fps: float, capturing: bool) -> None:
    """Draw on-screen instruction overlay with status information.
    
    Args:
        img: OpenCV image to draw on (modified in-place).
        sign: Current sign being collected.
        burst_mode: Whether in burst mode.
        frame_count: Current frame count.
        total_frames: Total frames captured in session.
        start_time: Session start time.
        fps: Current frames per second.
        capturing: Whether currently capturing.
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.6
    thickness = 2
    color_text = (255, 255, 255)
    color_bg = (0, 0, 0)
    y_offset = 30
    line_spacing = 35

    # Create semi-transparent overlay for text background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 10 + 240), color_bg, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Instructions
    y = y_offset
    cv2.putText(img, f"Sign: {sign.upper()}", (20, y), font, font_size, color_text, thickness)
    y += line_spacing
    
    if burst_mode:
        cv2.putText(img, "Mode: BURST (Hold 'S' to capture)", (20, y), font, font_size, color_text, thickness)
    else:
        cv2.putText(img, "Mode: MANUAL (Press sign key to capture)", (20, y), font, font_size, color_text, thickness)
    y += line_spacing

    # Capture status
    status_text = "STATUS: CAPTURING" if capturing else "Status: Ready"
    status_color = (0, 255, 0) if capturing else color_text
    cv2.putText(img, status_text, (20, y), font, font_size, status_color, thickness)
    y += line_spacing

    # Statistics
    duration = time.time() - start_time
    cv2.putText(img, f"Total Frames: {total_frames}", (20, y), font, font_size, color_text, thickness)
    y += line_spacing
    cv2.putText(img, f"Duration: {duration:.1f}s | FPS: {fps:.1f}", (20, y), font, font_size, color_text, thickness)
    y += line_spacing
    cv2.putText(img, "Press 'q' to quit | 's' to toggle burst", (20, y), font, font_size, (200, 200, 255), thickness)


def main():
    """Main data collection function with burst capture support."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect hand sign images from webcam.")
    parser.add_argument("--sign", type=str, default=None,
                        help="Specific sign to collect (overrides config). If not specified, uses first sign from config.")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
        create_directories(config)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return

    # Setup logging
    try:
        log_dir = config["paths"]["logs"]
        logger = setup_logger(__name__, log_dir=log_dir)
    except OSError as e:
        print(f"WARNING: Could not setup logging: {e}")
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    raw_root = Path(config["paths"]["raw_data"])
    config_signs = config["collection"]["signs"]
    
    # Determine which sign to collect
    if args.sign:
        sign = args.sign.upper()
        if sign not in [s.upper() for s in config_signs]:
            logger.error(f"Sign '{sign}' not in configured signs: {config_signs}")
            print(f"ERROR: Sign '{sign}' not in configured signs: {config_signs}")
            return
    else:
        sign = config_signs[0].upper()

    # Create sign-specific directory
    try:
        out_dir = raw_root / sign
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting data collection for sign: {sign}")
        logger.info(f"Output directory: {out_dir}")
    except OSError as e:
        logger.error(f"Failed to create directory {out_dir}: {e}")
        print(f"ERROR: Failed to create directory {out_dir}: {e}")
        return

    image_width = int(config["collection"]["image_width"])
    image_height = int(config["collection"]["image_height"])
    capture_interval = float(config["collection"]["capture_interval"])

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to initialize webcam. Please check camera connection.")
        print("ERROR: Failed to initialize webcam. Please check camera connection.")
        return

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

        # Collection state variables
        burst_mode = False
        prev_s_pressed = False
        last_saved = 0.0
        total_frames = 0
        session_start = time.time()
        frame_times = []
        burst_frame_index = 0

        logger.info(f"Collection started. Press 's' to toggle burst mode, 'q' to quit.")
        print(f"Collection for '{sign}' started.")
        print(f"Press 's' to toggle burst mode, 'q' to quit.")

        while True:
            success, img = cap.read()
            if not success:
                logger.warning("Failed to read frame from webcam.")
                break

            frame_times.append(time.time())
            if len(frame_times) > 30:  # Keep last 30 frames for FPS calculation
                frame_times.pop(0)
            
            current_fps = len(frame_times) / (frame_times[-1] - frame_times[0] + 1e-6)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info(f"User quit. Total frames captured: {total_frames}")
                break

            # Edge detection for 'S' key: detect transition from not pressed to pressed
            is_s_pressed = key == ord('s') or key == ord('S')
            
            # Toggle burst mode on 'S' key press edge (not pressed -> pressed)
            if is_s_pressed and not prev_s_pressed:
                burst_mode = not burst_mode
                logger.info(f"Burst mode: {'ON' if burst_mode else 'OFF'}")

            # Determine if currently capturing
            capturing = burst_mode and is_s_pressed

            # Burst mode logic: capture continuously while 'S' is held and burst mode is on
            if capturing:
                now = time.time()
                if now - last_saved >= capture_interval:
                    try:
                        saved = save_frame(img, out_dir, sign, burst_frame_index)
                        total_frames += 1
                        burst_frame_index += 1
                        last_saved = now
                        logger.debug(f"Saved frame {total_frames}: {saved.name}")
                    except Exception as e:
                        logger.error(f"Failed to save frame: {e}")
            else:
                burst_frame_index = 0  # Reset index when not capturing

            # Draw overlay
            draw_overlay(img, sign, burst_mode, burst_frame_index, total_frames,
                        session_start, current_fps, capturing)
            
            # Update previous state for next iteration
            prev_s_pressed = is_s_pressed

            cv2.imshow("Hand Sign Collection", img)

        # Log collection statistics
        session_duration = time.time() - session_start
        avg_fps = total_frames / (session_duration + 1e-6)
        logger.info(f"Collection session completed:")
        logger.info(f"  Total frames captured: {total_frames}")
        logger.info(f"  Session duration: {session_duration:.2f}s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")
        logger.info(f"  Output directory: {out_dir}")

        print(f"\nCollection Complete!")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {session_duration:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Saved to: {out_dir}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
