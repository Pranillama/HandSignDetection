"""Visualization utilities for hand landmarks and predictions."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

"""
Displays the sign label and confidence percentage 
"""
def display_prediction(
    frame: np.ndarray, label: str, confidence: float, threshold: float
) -> np.ndarray:

    org = (10, 40)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    line_type = cv2.LINE_AA

    if confidence < threshold:
        text = "Unknown"
        color = (255, 255, 255) 
    else:
        text = f"{label}: {confidence:.0%}"
        color = (0, 255, 0)  

    # Draw black shadow for legibility (shadow effect)
    cv2.putText(
        frame,
        text,
        org,
        font_face,
        font_scale,
        (0, 0, 0),  
        thickness=4,
        lineType=line_type,
    )
    # Draw the main text on top
    cv2.putText(
        frame,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness=thickness,
        lineType=line_type,
    )
    return frame

"""
Render FPS counter at the bottom-left of the frame.
"""
def display_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    
    text = f"FPS: {fps:.1f}"
    org = (10, frame.shape[0] - 10)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 255) 
    thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(
        frame,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness=thickness,
        lineType=line_type,
    )
    return frame