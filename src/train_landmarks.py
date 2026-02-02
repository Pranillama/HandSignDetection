"""Placeholder: Train a landmarks-based feedforward neural network.

This stub will later contain training logic for models using MediaPipe landmarks.
"""

from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    logger = setup_logger(__name__)
    config = load_config()
    logger.info("Train landmarks placeholder: implement training logic here.")


if __name__ == "__main__":
    main()
