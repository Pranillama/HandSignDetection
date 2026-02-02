"""Placeholder: Real-time inference using landmarks-based model.

This stub will later implement webcam inference using a landmarks model.
"""

from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    logger = setup_logger(__name__)
    config = load_config()
    logger.info("Inference landmarks placeholder: implement real-time inference here.")


if __name__ == "__main__":
    main()
