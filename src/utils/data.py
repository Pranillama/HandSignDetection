"""Data management utilities for ASL hand sign detection.

This module provides functions for loading, organizing, and splitting image data
for the hand sign detection pipeline. It handles image loading from directory
structures, stratified train/val/test splitting (without sklearn), and dataset
validation.

Key functions:
    - get_class_mapping(): Map sign labels to integer indices
    - load_images_from_directory(): Load images from sign subdirectories
    - stratified_split(): Manually split data maintaining class distribution
    - save_images_to_split(): Organize images into processed splits
    - validate_dataset_structure(): Check dataset integrity
    - count_images_per_class(): Count images per sign
    - check_class_balance(): Verify class distribution balance
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import shutil

from src.utils.config import load_config

# Module constants
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
DEFAULT_RANDOM_SEED = 42

logger = logging.getLogger(__name__)

# Load configuration at module level
try:
    _config = load_config()
    _SIGNS = _config["collection"]["signs"]
except Exception as exc:
    logger.warning("Failed to load config at module import: %s. Falling back to empty signs.", exc)
    _SIGNS = []


def get_class_mapping() -> Dict[str, int]:
    """Get mapping from sign labels to integer indices.
    
    Returns a dictionary where keys are sign labels (e.g., "A", "B", "C")
    and values are zero-indexed integers.
    """
    if not _SIGNS:
        logger.warning("No signs configured in config. Returning empty mapping.")
        return {}
    
    mapping = {sign.upper(): idx for idx, sign in enumerate(_SIGNS)}
    logger.debug(f"Class mapping: {mapping}")
    return mapping


def get_reverse_class_mapping() -> Dict[int, str]:
    """Get reverse mapping from integer indices to sign labels.
    
    Returns a dictionary where keys are zero-indexed integers and values
    are sign labels.
    """
    if not _SIGNS:
        logger.warning("No signs configured in config. Returning empty mapping.")
        return {}
    
    reverse_map = {idx: sign.upper() for idx, sign in enumerate(_SIGNS)}
    logger.debug(f"Reverse class mapping: {reverse_map}")
    return reverse_map


def load_images_from_directory(directory: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load images from a directory structure with sign subdirectories.
    
    Scans the directory for subdirectories matching configured signs. Each
    subdirectory should contain image files (.jpg, .jpeg, .png). Images are
    loaded in BGR format using cv2.imread.
    
    Args:
        directory: Path to root directory containing sign subdirectories.
    
    Returns:
        Tuple of (images, labels, file_paths) where:
        - images: List of numpy arrays (BGR format) with shape (H, W, 3)
        - labels: List of sign labels corresponding to images
        - file_paths: List of absolute file paths for each image
    """
    images: List[np.ndarray] = []
    labels: List[str] = []
    file_paths: List[str] = []
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return images, labels, file_paths
    
    if not dir_path.is_dir():
        logger.warning(f"Path is not a directory: {directory}")
        return images, labels, file_paths
    
    logger.info(f"Loading images from {directory}")
    
    # Get configured signs
    if not _SIGNS:
        logger.warning("No signs configured. Cannot load images.")
        return images, labels, file_paths
    
    image_count = 0
    skipped_count = 0
    
    # Iterate through sign subdirectories
    for sign in _SIGNS:
        sign_dir = dir_path / sign.upper()
        
        if not sign_dir.exists():
            logger.warning(f"Sign directory not found: {sign_dir}")
            continue
        
        if not sign_dir.is_dir():
            logger.warning(f"Sign path is not a directory: {sign_dir}")
            continue
        
        # Scan for image files
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            for img_file in sign_dir.glob(f"*{ext}"):
                try:
                    # Read image in BGR format
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        logger.warning(f"Failed to load image (returned None): {img_file}")
                        skipped_count += 1
                        continue
                    
                    # Validate image
                    if img.ndim != 3 or img.shape[2] != 3:
                        logger.warning(f"Invalid image shape {img.shape}: {img_file}")
                        skipped_count += 1
                        continue
                    
                    images.append(img)
                    labels.append(sign.upper())
                    file_paths.append(str(img_file.absolute()))
                    image_count += 1
                
                except Exception as exc:
                    logger.warning(f"Error loading image {img_file}: {exc}")
                    skipped_count += 1
    
    logger.info(f"Loaded {image_count} images, skipped {skipped_count}")
    return images, labels, file_paths


def stratified_split(
    images: List[np.ndarray],
    labels: List[str],
    file_paths: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Tuple[List[np.ndarray], List[str], List[str]]]:
    """Split data into train/val/test while maintaining class distribution.
    
    Uses "stratified sampling" to ensure each split maintains the same class
    distribution as the original dataset. Scikit-learn is not available, so
    this is implemented manually using numpy.
    
    Args:
        images: List of image arrays (BGR format).
        labels: List of class labels for each image.
        file_paths: List of file paths for each image.
        train_ratio: Fraction of data for training (default 0.7).
        val_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for testing (default 0.15).
        random_seed: Random seed for reproducibility (default 42).
    
    Returns:
        Dictionary with keys "train", "val", "test", each containing a tuple of
        (images, labels, file_paths) for that split.
    """
    # Validate inputs
    if len(images) != len(labels) or len(labels) != len(file_paths):
        raise ValueError("Images, labels, and file_paths must have same length")
    
    if len(images) == 0:
        logger.warning("Cannot split empty data")
        return {"train": ([], [], []), "val": ([], [], []), "test": ([], [], [])}
    
    # Validate ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=0.01):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
    
    logger.info(f"Stratified split: train={train_ratio:.2%}, val={val_ratio:.2%}, test={test_ratio:.2%}")
    
    # Group indices by label
    unique_labels = list(set(labels))
    label_indices: Dict[str, List[int]] = {label: [] for label in unique_labels}
    
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    
    # Split each label's indices
    rng = np.random.RandomState(random_seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    
    for label, indices in label_indices.items():
        # Shuffle indices for this label
        shuffled_indices = indices.copy()
        rng.shuffle(shuffled_indices)
        
        # Calculate split points
        n = len(shuffled_indices)
        train_split = int(n * train_ratio)
        val_split = train_split + int(n * val_ratio)
        
        # Assign to splits
        train_indices.extend(shuffled_indices[:train_split])
        val_indices.extend(shuffled_indices[train_split:val_split])
        test_indices.extend(shuffled_indices[val_split:])
        
        logger.debug(
            f"Label {label}: train={train_split}, val={val_split - train_split}, "
            f"test={n - val_split}"
        )
    
    # Extract data for each split
    def extract_split(indices: List[int]) -> Tuple[List[np.ndarray], List[str], List[str]]:
        split_images = [images[i] for i in indices]
        split_labels = [labels[i] for i in indices]
        split_paths = [file_paths[i] for i in indices]
        return split_images, split_labels, split_paths
    
    splits = {
        "train": extract_split(train_indices),
        "val": extract_split(val_indices),
        "test": extract_split(test_indices),
    }
    
    # Log statistics
    log_split_statistics(splits)
    
    return splits


def save_images_to_split(
    images: List[np.ndarray],
    labels: List[str],
    file_paths: List[str],
    output_dir: str,
    split_name: str,
) -> int:
    """Save images to organized split directory structure.
    """
    if len(images) == 0:
        logger.warning("No images to save")
        return 0
    
    if len(images) != len(labels) or len(labels) != len(file_paths):
        logger.error("Images, labels, and file_paths length mismatch")
        return 0
    
    output_path = Path(output_dir)
    
    # Validate output directory is writable
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Try to write a test file to verify permissions
        test_file = output_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as exc:
        logger.error(f"Output directory not writable: {output_dir}. Error: {exc}")
        return 0
    
    saved_count = 0
    
    logger.info(f"Saving {len(images)} images to {output_dir}/{split_name}")
    
    for image, label, src_path in zip(images, labels, file_paths):
        try:
            # Create sign subdirectory
            sign_dir = output_path / split_name / label.upper()
            sign_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract filename from source path
            filename = Path(src_path).name
            dest_path = sign_dir / filename
            
            # Copy file preserving metadata
            shutil.copy2(src_path, str(dest_path))
            saved_count += 1
        
        except Exception as exc:
            logger.warning(f"Failed to save image {src_path} to {dest_path}: {exc}")
    
    logger.info(f"Successfully saved {saved_count}/{len(images)} images to {split_name} split")
    return saved_count


def validate_dataset_structure(directory: str, expected_signs: Optional[List[str]] = None) -> bool:
    """Validate dataset directory structure.
    
    Validates the directory structure of either:
    - A single split directory with sign subdirectories: {directory}/{sign}/
    - A processed root with split subdirectories: {directory}/{split}/{sign}/
    
    Automatically detects the structure and validates accordingly. Checks that
    all expected sign subdirectories exist and contain at least one image file.
    """
    if expected_signs is None:
        expected_signs = _SIGNS
    
    if not expected_signs:
        logger.warning("No expected signs to validate")
        return False
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Dataset directory does not exist: {directory}")
        return False
    
    logger.info(f"Validating dataset structure in {directory}")
    
    # Detect structure type: check if subdirectories are splits or signs
    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
    split_names = {"train", "val", "test"}
    
    # Check if this looks like a processed root with splits
    has_splits = any(d.name.lower() in split_names for d in subdirs)
    has_signs = any(d.name.upper() in [s.upper() for s in expected_signs] for d in subdirs)
    
    all_valid = True
    
    if has_splits and not has_signs:
        # This is a processed root with split subdirectories
        logger.debug("Detected processed root structure with split subdirectories")
        
        for split_dir in subdirs:
            if split_dir.name.lower() not in split_names:
                continue
            
            logger.debug(f"Validating split: {split_dir.name}")
            
            # Validate signs within this split
            for sign in expected_signs:
                sign_dir = split_dir / sign.upper()
                
                if not sign_dir.exists():
                    logger.warning(f"Missing sign directory in {split_dir.name}: {sign_dir}")
                    all_valid = False
                    continue
                
                if not sign_dir.is_dir():
                    logger.warning(f"Sign path is not a directory: {sign_dir}")
                    all_valid = False
                    continue
                
                # Check for at least one image
                image_files = []
                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.extend(sign_dir.glob(f"*{ext}"))
                
                if not image_files:
                    logger.warning(f"No images found in {sign_dir}")
                    all_valid = False
                else:
                    logger.debug(f"Sign {sign} in {split_dir.name}: {len(image_files)} images")
    
    else:
        # This is a single split directory with sign subdirectories
        logger.debug("Detected single split structure with sign subdirectories")
        
        for sign in expected_signs:
            sign_dir = dir_path / sign.upper()
            
            if not sign_dir.exists():
                logger.warning(f"Missing sign directory: {sign_dir}")
                all_valid = False
                continue
            
            if not sign_dir.is_dir():
                logger.warning(f"Sign path is not a directory: {sign_dir}")
                all_valid = False
                continue
            
            # Check for at least one image
            image_files = []
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.extend(sign_dir.glob(f"*{ext}"))
            
            if not image_files:
                logger.warning(f"No images found in sign directory: {sign_dir}")
                all_valid = False
            else:
                logger.debug(f"Sign {sign}: {len(image_files)} images found")
    
    if all_valid:
        logger.info("Dataset structure validation passed")
    else:
        logger.warning("Dataset structure validation failed")
    
    return all_valid


def count_images_per_class(directory: str) -> Dict[str, int]:
    """Count images in each sign subdirectory.
    """
    counts: Dict[str, int] = {}
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return counts
    
    for sign in _SIGNS:
        sign_dir = dir_path / sign.upper()
        
        if not sign_dir.exists():
            counts[sign.upper()] = 0
            continue
        
        image_files = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(sign_dir.glob(f"*{ext}"))
        
        counts[sign.upper()] = len(image_files)
    
    logger.debug(f"Image counts per class: {counts}")
    return counts


def check_class_balance(counts: Dict[str, int], threshold: float = 0.3) -> bool:
    """Check if class distribution is balanced within threshold.
    
    Calculates the ratio of minimum to maximum class counts. If ratio is
    greater than (1 - threshold), classes are considered balanced.
    """
    if not counts or len(counts) < 2:
        logger.warning("Cannot check balance with less than 2 classes")
        return False
    
    values = list(counts.values())
    min_count = min(values)
    max_count = max(values)
    
    if max_count == 0:
        logger.warning("All classes have zero images")
        return False
    
    ratio = min_count / max_count
    is_balanced = ratio > (1.0 - threshold)
    
    logger.info(f"Class balance: min={min_count}, max={max_count}, ratio={ratio:.2%}")
    
    if not is_balanced:
        logger.warning(
            f"Class imbalance detected: ratio {ratio:.2%} below threshold "
            f"{(1.0 - threshold):.2%}"
        )
    else:
        logger.info("Classes are well balanced")
    
    return is_balanced


def get_split_ratios_from_config(config_path: str = "config/config.yaml") -> Tuple[float, float, float]:
    """Load train/val/test split ratios from configuration.
    Returns:
        Tuple of (train_ratio, val_ratio, test_ratio).
    """
    try:
        config = load_config(config_path)
        train_ratio = float(config["preprocessing"]["train_split"])
        val_ratio = float(config["preprocessing"]["val_split"])
        test_ratio = float(config["preprocessing"]["test_split"])
        
        ratio_sum = train_ratio + val_ratio + test_ratio
        if not np.isclose(ratio_sum, 1.0, atol=0.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
        
        logger.debug(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        return train_ratio, val_ratio, test_ratio
    
    except Exception as exc:
        logger.error(f"Failed to load split ratios from config: {exc}")
        raise


def log_split_statistics(splits: Dict[str, Tuple[List, List, List]]) -> None:
    """Log detailed statistics for each split.
    Logs total count and per-class distribution for train, val, and test splits.
    """
    for split_name, (split_images, split_labels, _) in splits.items():
        if not split_images:
            logger.info(f"{split_name.upper()} split: 0 images")
            continue
        
        total = len(split_images)
        
        # Count per class
        class_counts = {}
        for label in split_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        logger.info(f"{split_name.upper()} split: {total} images")
        for label, count in sorted(class_counts.items()):
            percentage = (count / total) * 100
            logger.info(f"  {label}: {count} images ({percentage:.1f}%)")
