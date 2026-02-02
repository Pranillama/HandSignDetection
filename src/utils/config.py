"""Configuration loader for ASL hand sign detection.

Provides load_config(), validate_config(), and create_directories() helpers to
centralize YAML-based configuration loading and validation.
"""

from pathlib import Path
import os
import yaml
import warnings
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and validate YAML configuration from the given path.

    Args:
        config_path: Relative or absolute path to the YAML configuration file.

    Returns:
        A validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        yaml.YAMLError: If parsing the YAML fails.
        ValueError: If validation fails.
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. Please create config/config.yaml and run from project root."
        )

    try:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in configuration file: {e}")

    if not isinstance(cfg, dict):
        raise ValueError("Configuration file did not contain a mapping at top level.")

    validate_config(cfg)
    return cfg


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary and raise ValueError on issues.

    Checks presence of required keys, basic types, ranges, and logical constraints.
    """
    required_top = [
        "paths",
        "collection",
        "preprocessing",
        "training_landmarks",
        "training_cnn",
        "inference",
    ]

    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Paths
    paths_required = [
        "raw_data",
        "processed_data",
        "landmarks_data",
        "models_landmarks",
        "models_cnn",
        "logs",
        "results",
    ]
    paths = config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Invalid configuration for 'paths': expected a mapping of path names to strings")
    for key in paths_required:
        if key not in paths:
            raise ValueError(f"Missing required configuration key: paths.{key}")
        if not isinstance(paths[key], str) or not paths[key].strip():
            raise ValueError(f"Invalid value for paths.{key}: expected non-empty string, got {repr(paths[key])}")
        # Ensure parent directory is creatable/resolvable
        try:
            p = Path(paths[key])
            parent = p.parent if p.parent != Path('') else Path('.')
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Invalid path for paths.{key}: cannot ensure parent directory for {paths[key]}: {e}")

    # Collection
    coll = config["collection"]
    coll_req = ["image_width", "image_height", "capture_interval", "signs"]
    for key in coll_req:
        if key not in coll:
            raise ValueError(f"Missing required configuration key: collection.{key}")

    if not isinstance(coll["image_width"], int) or coll["image_width"] <= 0:
        raise ValueError(f"Invalid value for collection.image_width: expected positive int, got {coll['image_width']}")
    if not isinstance(coll["image_height"], int) or coll["image_height"] <= 0:
        raise ValueError(f"Invalid value for collection.image_height: expected positive int, got {coll['image_height']}")
    if not isinstance(coll["capture_interval"], (int, float)) or coll["capture_interval"] < 0:
        raise ValueError(f"Invalid value for collection.capture_interval: expected non-negative number, got {coll['capture_interval']}")
    if not isinstance(coll["signs"], list) or not all(isinstance(s, str) for s in coll["signs"]):
        raise ValueError(f"Invalid value for collection.signs: expected list of strings, got {coll['signs']}")

    # Preprocessing splits and params
    pre = config["preprocessing"]
    pre_req = ["train_split", "val_split", "test_split", "min_detection_confidence"]
    for key in pre_req:
        if key not in pre:
            raise ValueError(f"Missing required configuration key: preprocessing.{key}")

    splits = (pre["train_split"], pre["val_split"], pre["test_split"])  # type: ignore
    if not all(isinstance(x, (int, float)) for x in splits):
        raise ValueError("Invalid split ratios: expected numeric values for train/val/test splits.")

    ssum = sum(float(x) for x in splits)
    if abs(ssum - 1.0) > 0.01:
        raise ValueError(f"Invalid split ratios: sum is {ssum}, expected 1.0")

    if not (0 <= pre["min_detection_confidence"] <= 1):
        raise ValueError("Invalid preprocessing.min_detection_confidence: expected value between 0 and 1")

    # Training landmarks
    t_land = config["training_landmarks"]
    tland_req = ["epochs", "batch_size", "learning_rate", "hidden_layers", "dropout_rate"]
    for key in tland_req:
        if key not in t_land:
            raise ValueError(f"Missing required configuration key: training_landmarks.{key}")

    if not isinstance(t_land["epochs"], int) or t_land["epochs"] <= 0:
        raise ValueError("Invalid value for training_landmarks.epochs: expected positive integer")
    if not isinstance(t_land["batch_size"], int) or t_land["batch_size"] <= 0:
        raise ValueError("Invalid value for training_landmarks.batch_size: expected positive integer")
    if not isinstance(t_land["learning_rate"], (int, float)) or t_land["learning_rate"] <= 0:
        raise ValueError("Invalid value for training_landmarks.learning_rate: expected positive number")
    if not isinstance(t_land["hidden_layers"], list) or not all(isinstance(x, int) and x > 0 for x in t_land["hidden_layers"]):
        raise ValueError("Invalid value for training_landmarks.hidden_layers: expected list of positive integers")
    if not isinstance(t_land["dropout_rate"], (int, float)) or not (0 <= t_land["dropout_rate"] <= 1):
        raise ValueError("Invalid value for training_landmarks.dropout_rate: expected between 0 and 1")

    # Training CNN
    t_cnn = config["training_cnn"]
    tcnn_req = ["epochs", "batch_size", "learning_rate", "image_size", "architecture"]
    for key in tcnn_req:
        if key not in t_cnn:
            raise ValueError(f"Missing required configuration key: training_cnn.{key}")

    if not isinstance(t_cnn["image_size"], list) or len(t_cnn["image_size"]) != 2:
        raise ValueError("Invalid value for training_cnn.image_size: expected list [width, height]")
    if not all(isinstance(x, int) and x > 0 for x in t_cnn["image_size"]):
        raise ValueError("Invalid value for training_cnn.image_size: expected positive integers")

    if not isinstance(t_cnn["epochs"], int) or t_cnn["epochs"] <= 0:
        raise ValueError("Invalid value for training_cnn.epochs: expected positive integer")
    if not isinstance(t_cnn["batch_size"], int) or t_cnn["batch_size"] <= 0:
        raise ValueError("Invalid value for training_cnn.batch_size: expected positive integer")
    if not isinstance(t_cnn["learning_rate"], (int, float)) or t_cnn["learning_rate"] <= 0:
        raise ValueError("Invalid value for training_cnn.learning_rate: expected positive number")
    if not isinstance(t_cnn["architecture"], str) or not t_cnn["architecture"].strip():
        raise ValueError("Invalid value for training_cnn.architecture: expected non-empty string")

    # Inference
    inf = config["inference"]
    if "confidence_threshold" not in inf:
        raise ValueError("Missing required configuration key: inference.confidence_threshold")
    if "display_fps" not in inf:
        raise ValueError("Missing required configuration key: inference.display_fps")

    if not (0 <= inf["confidence_threshold"] <= 1):
        raise ValueError("Invalid value for inference.confidence_threshold: expected between 0 and 1")
    if not isinstance(inf["display_fps"], bool):
        raise ValueError("Invalid value for inference.display_fps: expected boolean")


def create_directories(config: Dict[str, Any]) -> None:
    """Create directories listed in the config['paths'] mapping.

    Any OSError raised while creating directories will be caught and a warning will be emitted.
    """
    paths = config.get("paths", {})
    for key, rel in paths.items():
        try:
            Path(rel).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warnings.warn(f"Could not create directory for paths.{key} at {rel}: {e}")
