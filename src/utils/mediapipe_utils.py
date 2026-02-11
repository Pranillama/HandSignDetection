"""MediaPipe hand landmark detection utilities.

This module provides a lightweight wrapper around MediaPipe's Hands
solution for extracting hand landmarks from BGR (OpenCV) images. It
includes a reusable ``HandDetector`` class and a set of helper functions
for extracting, normalizing, and validating landmark data.

"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp

import numpy as np

from src.utils.config import load_config

# Module constants
NUM_HAND_LANDMARKS = 21
LANDMARK_FEATURES = 3
FLATTENED_SIZE = NUM_HAND_LANDMARKS * LANDMARK_FEATURES

logger = logging.getLogger(__name__)

# Load default min_detection_confidence from config
try:
    _config = load_config()
    _DEFAULT_MIN_DETECTION_CONFIDENCE = float(_config["preprocessing"]["min_detection_confidence"])
except Exception as exc:
    logger.warning("Failed to load config for default min_detection_confidence: %s. Using 0.5.", exc)
    _DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5


class HandDetector:
    """Wrapper for MediaPipe Hands.
    
    The detector expects BGR images (OpenCV convention). MediaPipe requires
    RGB input, so conversion is handled internally.
    
    Args:
        min_detection_confidence: Minimum confidence for detection (0-1).
            Defaults to config value.
        min_tracking_confidence: Minimum confidence for tracking (0-1).
        max_num_hands: Maximum number of hands to detect.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = _DEFAULT_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
    ) -> None:
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1")
        if max_num_hands < 1:
            raise ValueError("max_num_hands must be >= 1")
        
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.max_num_hands = int(max_num_hands)
        
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - initialization error
            logger.exception("Failed to initialize MediaPipe Hands: %s", exc)
            raise
    
    def detect(self, image: np.ndarray) -> Optional[mp.solutions.hands.HandsResults]:
        """Run hand detection on a BGR image and return MediaPipe results.
        
        Args:
            image: BGR image as numpy array with shape (H, W, 3) and dtype uint8.
        
        Returns:
            MediaPipe HandsResults on success, or ``None`` on failure.
        """
        if not isinstance(image, np.ndarray):
            logger.warning("detect() expected numpy.ndarray, got %s", type(image))
            return None
        
        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning("detect() expected image with shape (H, W, 3), got %s", getattr(image, "shape", None))
            return None
        
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self._hands is None:
                logger.error("HandDetector used after being closed")
                return None
            results = self._hands.process(rgb)
            return results
        except Exception as exc:
            logger.exception("Error during hand detection: %s", exc)
            return None
    
    def close(self) -> None:
        """Release MediaPipe resources.
        Safe to call multiple times.
        """
        try:
            if hasattr(self, "_hands") and self._hands is not None:
                self._hands.close()
                self._hands = None
        except Exception:
            logger.exception("Error closing MediaPipe Hands instance")
    
    def __enter__(self) -> "HandDetector":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def landmarks_to_array(hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
    """Convert MediaPipe NormalizedLandmarkList to a (21, 3) numpy array.
    
    Returns:
        Array of shape (21, 3) with dtype float32 containing [x, y, z].
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append((lm.x, lm.y, lm.z))
    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape != (NUM_HAND_LANDMARKS, LANDMARK_FEATURES):
        logger.warning("landmarks_to_array produced unexpected shape %s", arr.shape)
    return arr


def extract_landmarks(image: np.ndarray, detector: Optional[HandDetector] = None) -> Optional[np.ndarray]:
    """Detect and extract the first hand's landmarks as a (21, 3) array.
    
    If ``detector`` is not provided a temporary ``HandDetector`` is created
    with config-based min_detection_confidence and closed after use.
    """
    if not isinstance(image, np.ndarray):
        logger.warning("extract_landmarks expects a numpy.ndarray image")
        return None
    
    temp = False
    if detector is None:
        detector = HandDetector()
        temp = True
    
    try:
        results = detector.detect(image)
        if not results or not getattr(results, "multi_hand_landmarks", None):
            return None
    
        first = results.multi_hand_landmarks[0]
        return landmarks_to_array(first)
    except Exception:
        logger.exception("Failed to extract landmarks")
        return None
    finally:
        if temp:
            detector.close()


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmark array and return flattened vector of length 63.
    
    - Accepts shape (21, 3) or (63,) and returns flattened (63,) array.
    - Clips x and y to [0, 1] (MediaPipe uses normalized coordinates).
    - Normalizes z via min-max scaling per hand to [0, 1].
    """
    if not isinstance(landmarks, np.ndarray):
        raise ValueError("landmarks must be a numpy array")
    
    if landmarks.shape == (FLATTENED_SIZE,):
        landmarks = landmarks.reshape(NUM_HAND_LANDMARKS, LANDMARK_FEATURES)
    elif landmarks.shape != (NUM_HAND_LANDMARKS, LANDMARK_FEATURES):
        raise ValueError(f"Invalid landmarks shape {landmarks.shape}, expected (21,3) or (63,) ")
    
    lm = landmarks.astype(np.float32).copy()
    # Clamp x and y
    lm[:, 0] = np.clip(lm[:, 0], 0.0, 1.0)
    lm[:, 1] = np.clip(lm[:, 1], 0.0, 1.0)
    
    # Normalize z via min-max per hand
    z = lm[:, 2]
    if np.allclose(z, 0.0):
        lm[:, 2] = 0.0
    else:
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        if zmax - zmin > 1e-6:
            lm[:, 2] = (z - zmin) / (zmax - zmin)
        else:
            lm[:, 2] = 0.0
    
    lm[:, 2] = np.clip(lm[:, 2], 0.0, 1.0)
    return lm.reshape(-1)


def is_hand_detected(image: np.ndarray, min_detection_confidence: float = _DEFAULT_MIN_DETECTION_CONFIDENCE) -> bool:
    """Return True if at least one hand is detected in the image.
    
    This function creates a temporary detector with the provided confidence
    (defaults to config value) and closes it before returning.
    """
    if not isinstance(image, np.ndarray):
        logger.warning("is_hand_detected expected numpy.ndarray image")
        return False
    
    detector = None
    try:
        detector = HandDetector(min_detection_confidence=min_detection_confidence)
        results = detector.detect(image)
        return bool(results and getattr(results, "multi_hand_landmarks", None) and len(results.multi_hand_landmarks) > 0)
    except Exception:
        logger.exception("Error in is_hand_detected")
        return False
    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass

