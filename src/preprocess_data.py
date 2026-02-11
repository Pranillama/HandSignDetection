"""Preprocessing pipeline for validating images and extracting landmarks.

This module orchestrates the complete preprocessing workflow:
1. Load raw images from source directory
2. Validate images contain detectable hands using MediaPipe
3. Perform stratified train/val/test split
4. Save validated images to organized processed directories
5. Extract and normalize hand landmarks
6. Save landmark arrays for training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import load_config, create_directories
from src.utils.logger import setup_logger
from src.utils.mediapipe_utils import (
    HandDetector,
    extract_landmarks,
    is_hand_detected,
    normalize_landmarks,
)
from src.utils.data import (
    get_class_mapping,
    get_split_ratios_from_config,
    load_images_from_directory,
    save_images_to_split,
    stratified_split,
    validate_dataset_structure,
)

logger = logging.getLogger(__name__)


def validate_images_with_hand_detection(
    images: List[np.ndarray],
    labels: List[str],
    file_paths: List[str],
    min_detection_confidence: float = 0.5,
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Validate images by detecting hands using MediaPipe.
    
    Filters out images where no hand is detected. Uses a single HandDetector with
    the provided confidence threshold to validate each image, avoiding per-image
    re-initialization overhead.
    
    Returns:
        Tuple of (valid_images, valid_labels, valid_paths) filtered to include
        only images with detected hands.
    """
    if not images:
        logger.warning("No images to validate")
        return [], [], []

    if len(images) != len(labels) or len(labels) != len(file_paths):
        logger.error("Images, labels, and file_paths length mismatch")
        return [], [], []

    logger.info(f"Validating {len(images)} images for hand detection")

    valid_images: List[np.ndarray] = []
    valid_labels: List[str] = []
    valid_paths: List[str] = []
    rejected_count = 0

    with HandDetector(min_detection_confidence=min_detection_confidence) as detector:
        for idx, (image, label, path) in enumerate(zip(images, labels, file_paths)):
            results = detector.detect(image)
            if results and getattr(results, "multi_hand_landmarks", None):
                valid_images.append(image)
                valid_labels.append(label)
                valid_paths.append(path)
            else:
                rejected_count += 1
                logger.debug(f"No hand detected in image: {path}")

    rejection_rate = (rejected_count / len(images)) * 100 if images else 0.0

    logger.info(
        f"Hand detection validation complete: "
        f"total={len(images)}, valid={len(valid_images)}, "
        f"rejected={rejected_count}, rejection_rate={rejection_rate:.1f}%"
    )

    return valid_images, valid_labels, valid_paths


def extract_and_save_landmarks(
    split_name: str,
    images: List[np.ndarray],
    labels: List[str],
    output_dir: str,
    detector: HandDetector,
) -> int:
    """Extract landmarks from images and save as NumPy arrays.
    
    Processes images to extract hand landmarks, normalizes them, and saves
    as NumPy arrays along with corresponding integer labels.
    
    Returns:
        Count of successfully processed images.
    """
    if not images:
        logger.warning(f"No images to process for {split_name} split")
        return 0

    if len(images) != len(labels):
        logger.error("Images and labels length mismatch")
        return 0

    logger.info(f"Extracting landmarks from {len(images)} images for {split_name} split")

    class_mapping = get_class_mapping()
    landmarks_list: List[np.ndarray] = []
    labels_int: List[int] = []
    failed_count = 0

    for idx, (image, label) in enumerate(zip(images, labels)):
        try:
            # Extract landmarks
            landmarks = extract_landmarks(image, detector=detector)
            if landmarks is None:
                failed_count += 1
                logger.debug(f"Failed to extract landmarks from image {idx} ({label})")
                continue

            # Normalize landmarks
            normalized = normalize_landmarks(landmarks)
            landmarks_list.append(normalized)
            labels_int.append(class_mapping[label.upper()])

        except Exception as exc:
            failed_count += 1
            logger.warning(f"Error processing image {idx}: {exc}")

    if not landmarks_list:
        logger.warning(f"No landmarks successfully extracted for {split_name} split")
        return 0

    # Save landmarks and labels as NumPy arrays
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        labels_array = np.array(labels_int, dtype=np.int32)

        landmarks_file = output_path / f"{split_name}.npy"
        labels_file = output_path / f"{split_name}_labels.npy"

        np.save(str(landmarks_file), landmarks_array)
        np.save(str(labels_file), labels_array)

        logger.info(
            f"Saved landmark arrays for {split_name}: "
            f"landmarks shape {landmarks_array.shape}, labels shape {labels_array.shape}"
        )
        logger.debug(f"  Landmarks: {landmarks_file}")
        logger.debug(f"  Labels: {labels_file}")

        success_count = len(landmarks_list)
        logger.info(
            f"Landmark extraction complete for {split_name}: "
            f"successful={success_count}, failed={failed_count}"
        )

        return success_count

    except Exception as exc:
        logger.error(f"Failed to save landmark arrays: {exc}")
        return 0


def main() -> int:
    """Runing the complete preprocessing pipeline.
    
    Returns:
        0 on success, 1 on failure.
    """
    # Setup and initialization
    logger = setup_logger(__name__)
    logger.info("=" * 70)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 70)

    try:
        config = load_config()
        create_directories(config)

        raw_data_path = config["paths"]["raw_data"]
        processed_data_path = config["paths"]["processed_data"]
        landmarks_data_path = config["paths"]["landmarks_data"]
        min_detection_confidence = float(config["preprocessing"]["min_detection_confidence"])

        logger.info("Configuration loaded successfully")
        logger.info(f"  Raw data path: {raw_data_path}")
        logger.info(f"  Processed data path: {processed_data_path}")
        logger.info(f"  Landmarks data path: {landmarks_data_path}")
        logger.info(f"  Min detection confidence: {min_detection_confidence}")

    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        return 1

    # Step 1: Load raw images
    logger.info("-" * 70)
    logger.info("Step 1: Loading raw images")
    logger.info("-" * 70)

    try:
        images, labels, file_paths = load_images_from_directory(raw_data_path)

        if not images:
            logger.error(f"No images found in {raw_data_path}")
            return 1

        logger.info(f"Loaded {len(images)} total images from {len(set(labels))} signs")
        for sign in sorted(set(labels)):
            count = sum(1 for l in labels if l == sign)
            logger.info(f"  {sign}: {count} images")

    except Exception as exc:
        logger.error(f"Failed to load images: {exc}")
        return 1

    # Step 2: Validate images with hand detection
    logger.info("-" * 70)
    logger.info("Step 2: Validating images with hand detection")
    logger.info("-" * 70)

    try:
        valid_images, valid_labels, valid_paths = validate_images_with_hand_detection(
            images, labels, file_paths, min_detection_confidence=min_detection_confidence
        )

        if not valid_images:
            logger.error("No valid images after hand detection validation")
            return 1

        logger.info(f"Validation passed: {len(valid_images)} valid images")

    except Exception as exc:
        logger.error(f"Hand detection validation failed: {exc}")
        return 1

    # Step 3: Perform stratified split
    logger.info("-" * 70)
    logger.info("Step 3: Performing stratified train/val/test split")
    logger.info("-" * 70)

    try:
        train_ratio, val_ratio, test_ratio = get_split_ratios_from_config()
        splits = stratified_split(
            valid_images,
            valid_labels,
            valid_paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        for split_name in ["train", "val", "test"]:
            split_images, split_labels, _ = splits[split_name]
            if split_images:
                logger.info(f"{split_name.upper()} split: {len(split_images)} images")
            else:
                logger.warning(f"{split_name.upper()} split: 0 images")

    except Exception as exc:
        logger.error(f"Stratified split failed: {exc}")
        return 1

    # Step 4: Save processed images
    logger.info("-" * 70)
    logger.info("Step 4: Saving processed images to split directories")
    logger.info("-" * 70)

    try:
        total_saved = 0
        for split_name in ["train", "val", "test"]:
            split_images, split_labels, split_paths = splits[split_name]
            saved_count = save_images_to_split(
                split_images, split_labels, split_paths, processed_data_path, split_name
            )
            total_saved += saved_count
            logger.info(f"{split_name.upper()}: saved {saved_count} images")

        logger.info(f"Total images saved: {total_saved}")

    except Exception as exc:
        logger.error(f"Failed to save processed images: {exc}")
        return 1

    # Step 5: Validate processed directory structure
    logger.info("-" * 70)
    logger.info("Step 5: Validating processed directory structure")
    logger.info("-" * 70)

    try:
        is_valid = validate_dataset_structure(processed_data_path)
        if not is_valid:
            logger.warning("Processed dataset structure validation failed")
        else:
            logger.info("Processed dataset structure validation passed")

    except Exception as exc:
        logger.error(f"Dataset structure validation failed: {exc}")
        return 1

    # Step 6: Extract and save landmarks
    logger.info("-" * 70)
    logger.info("Step 6: Extracting and saving landmarks")
    logger.info("-" * 70)

    landmarks_output_dir = Path(landmarks_data_path)

    try:
        total_landmarks = 0

        with HandDetector(min_detection_confidence=min_detection_confidence) as detector:
            for split_name in ["train", "val", "test"]:
                split_images, split_labels, _ = splits[split_name]

                if not split_images:
                    logger.info(f"Skipping landmark extraction for empty {split_name} split")
                    continue

                landmarks_count = extract_and_save_landmarks(
                    split_name,
                    split_images,
                    split_labels,
                    str(landmarks_output_dir),
                    detector,
                )
                total_landmarks += landmarks_count

        logger.info(f"Total landmarks extracted: {total_landmarks}")

    except Exception as exc:
        logger.error(f"Landmark extraction failed: {exc}")
        return 1

    # Step 7: Final summary and statistics
    logger.info("-" * 70)
    logger.info("Step 7: Final Summary")
    logger.info("-" * 70)

    try:
        logger.info("Pipeline Statistics:")
        logger.info(f"  Raw images scanned: {len(images)}")
        logger.info(f"  Valid images after hand detection: {len(valid_images)}")
        logger.info(
            f"  Images per split: "
            f"train={len(splits['train'][0])}, "
            f"val={len(splits['val'][0])}, "
            f"test={len(splits['test'][0])}"
        )
        logger.info(f"  Landmarks extracted: {total_landmarks}")

        # Verify output files
        landmarks_path = Path(landmarks_data_path)
        expected_files = ["train.npy", "train_labels.npy", "val.npy", "val_labels.npy", "test.npy", "test_labels.npy"]
        found_files = [f for f in expected_files if (landmarks_path / f).exists()]

        logger.info(f"  Landmark files created: {len(found_files)}/{len(expected_files)}")
        for f in found_files:
            file_path = landmarks_path / f
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"    {f}: {size_mb:.2f} MB")

        overall_success_rate = (len(valid_images) / len(images)) * 100 if images else 0.0
        logger.info(f"  Overall success rate: {overall_success_rate:.1f}%")

        logger.info("=" * 70)
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info("=" * 70)

        return 0

    except Exception as exc:
        logger.error(f"Final summary failed: {exc}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
