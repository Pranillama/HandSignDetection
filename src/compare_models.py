"""Placeholder: Compare landmarks and CNN models.

This stub will later implement systematic model comparison and reporting.
"""

from src.utils.logger import setup_logger
from src.utils.config import load_config


def main():
    logger = setup_logger(__name__)
    config = load_config()
    logger.info("Compare models placeholder: implement comparison and reporting here.")


if __name__ == "__main__":
    main()
