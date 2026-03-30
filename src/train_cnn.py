import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam  # type: ignore
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.config import load_config, create_directories
from src.utils.metrics import calculate_metrics, calculate_accuracy


class ASLImageDataset(Dataset):

    def __init__(self, root_dir: Path, class_names: list, transform):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label_idx, sign in enumerate(class_names):
            sign_dir = root_dir / sign
            if not sign_dir.is_dir():
                continue
            for img_path in sign_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label

"""
1. Simple Light weight CNN model.
"""
class LightweightCNN(nn.Module):

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 128),  # 8x8x128 = 8192 for 64x64 input
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.classifier(self.features(x))


def main():
    logger = setup_logger(__name__)
    config = load_config()
    create_directories(config)

    # read config
    tc = config["training_cnn"]
    epochs = tc["epochs"]
    batch_size = tc["batch_size"]
    lr = tc["learning_rate"]
    image_size = tc["image_size"]
    if "dropout_rate" not in tc:
        logger.warning("dropout_rate not found in config; using default dropout_rate=0.5")
    dropout_rate = tc.get("dropout_rate", 0.5)
    assert 0 <= dropout_rate <= 1, "Invalid dropout_rate: must be between 0 and 1"

    processed_data_dir = Path(config["paths"]["processed_data"])
    models_dir = Path(config["paths"]["models_cnn"])
    class_names = config["collection"]["signs"]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # build transform
    h, w = image_size
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    """
    2. load datasets
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = ASLImageDataset(processed_data_dir / split, class_names, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        datasets[split] = loader
        logger.info(f"Loaded {len(ds)} samples for {split}")

    """
    3. instantiate model
    """
    model = LightweightCNN(num_classes, dropout_rate).to(device)

    """ loss and optimizer"""
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    """
    4. training loop
    """
    train_acc_history: list[float] = []
    val_acc_history: list[float] = []
    
    # Early stopping setup 
    patience = 5 # stop if val_acc doesn't improve for 5 epochs
    best_val_acc = -1.0
    patience_counter = 0

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        # training phase
        model.train()
        correct = 0
        total = 0
        for xb, yb in datasets["train"]: #mini batch
            xb = xb.to(device)
            yb = yb.to(device)

            # forward pass
            logits = model(xb)
            # loss calculation
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            # back propagation and filter update (learnable parameters)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total if total > 0 else 0.0
        train_acc_history.append(train_acc)

        # validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in datasets["val"]:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total if total > 0 else 0.0
        val_acc_history.append(val_acc)

        logger.info(
            f"Epoch {epoch}/{epochs} — train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
        )

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)"
                )
                break
    training_time = time.time() - train_start

    """
    5. test evaluation
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    test_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in datasets["test"]:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size_actual = xb.size(0)
            test_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            preds = torch.argmax(logits, dim=1)
            #need cpu for accuracy (for eg for confusion matrix)
            all_preds.extend(preds.cpu().tolist()) #. extend: wraps all batched into single one long list
            all_labels.extend(yb.cpu().tolist())
    test_loss = test_loss / total_samples if total_samples > 0 else 0.0 # average loss per sample

    test_accuracy = calculate_accuracy(all_labels, all_preds)
    metrics_dict = calculate_metrics(all_labels, all_preds, class_names)

    # save model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"model_{ts}.pth"
    torch.save(model.state_dict(), model_path)

    # save metrics
    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "cnn",
        "training": {
            "epochs": epochs,
            "final_train_accuracy": train_acc_history[-1] if train_acc_history else None,
            "final_val_accuracy": val_acc_history[-1] if val_acc_history else None,
            "training_time_seconds": training_time,
        },
        "evaluation": {
            "test_accuracy": metrics_dict["accuracy"],
            "test_loss": test_loss,
            "confusion_matrix": metrics_dict["confusion_matrix"],
            "per_class_accuracy": metrics_dict["per_class_accuracy"],
        },
        "config": {
            "learning_rate": lr,
            "batch_size": batch_size,
            "image_size": image_size,
        },
    }
    metrics_path = models_dir / f"metrics_{ts}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # update symlink
    symlink_path = models_dir / "model_latest.pth"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(model_path.name)

    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
