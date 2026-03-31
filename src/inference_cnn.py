"""Real-time inference using CNN model.
"""

import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.train_cnn import LightweightCNN
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.visualization import display_prediction, display_fps


def main():
    logger = setup_logger(__name__)
    config = load_config()

    # Load model
    tc = config["training_cnn"]
    image_size = tc["image_size"]
    dropout_rate = tc.get("dropout_rate", 0.5)
    class_names = config["collection"]["signs"]
    num_classes = len(class_names)

    model_path = Path(config["paths"]["models_cnn"]) / "model_latest.pth"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Train a model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = LightweightCNN(num_classes=num_classes, dropout_rate=dropout_rate).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded model from {model_path}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam.")
        return

    threshold = config["inference"]["confidence_threshold"]
    show_fps = config["inference"]["display_fps"]

    # Build preprocessing transform
    h, w = image_size
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Session tracking
    session_start = time.time()
    frame_count = 0
    total_predictions = 0
    fps_sum = 0.0

    logger.info("Starting real-time inference. Press 'q' to quit.")

    try:
        # Per-frame inference loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tensor: torch.Tensor = transform(pil_img)  # type: ignore[assignment]
            tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)  # (1, num_classes)

            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)
            label = class_names[pred_idx.item()]
            conf_val = confidence.item()
            total_predictions += 1

            display_prediction(frame, label, conf_val, threshold)

            fps = 1.0 / (time.time() - frame_start + 1e-9)
            fps_sum += fps
            frame_count += 1

            if show_fps:
                display_fps(frame, fps)

            cv2.imshow("Hand Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    # Session logging
    session_duration = time.time() - session_start
    avg_fps = fps_sum / frame_count if frame_count > 0 else 0.0

    logger.info(f"Session duration: {session_duration:.2f} seconds")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info(f"Total predictions made: {total_predictions}")


if __name__ == "__main__":
    main()
