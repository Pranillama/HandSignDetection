import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam # type: ignore
from torch.utils.data import TensorDataset, DataLoader

from src.utils.logger import setup_logger
from src.utils.config import load_config, create_directories
from src.utils.metrics import calculate_metrics, calculate_accuracy

"""
1. feedforward neural network network(define model)
"""
class LandmarkNet(nn.Module):

    def __init__(
        self,
        input_size: int, # 63 
        hidden_layers: list,
        dropout_rate: float,
        num_classes: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) #randomly shuts off some neurons during training 
            in_features = size
        layers.append(nn.Linear(in_features, num_classes)) # final output layer
        self.network = nn.Sequential(*layers) # wraping all the above layers in sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


def main():
    logger = setup_logger(__name__)
    config = load_config()
    create_directories(config)

    #get info from config
    tl = config["training_landmarks"]
    epochs = tl["epochs"]
    batch_size = tl["batch_size"]
    lr = tl["learning_rate"]
    hidden_layers = tl["hidden_layers"]
    dropout_rate = tl["dropout_rate"]

    #get data 
    landmarks_dir = Path(config["paths"]["landmarks_data"])
    models_dir = Path(config["paths"]["models_landmarks"])
    class_names = config["collection"]["signs"]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    """
    2. load datasets
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        data_path = landmarks_dir / f"{split}.npy"
        labels_path = landmarks_dir / f"{split}_labels.npy"
        arr = np.load(data_path)
        lbl = np.load(labels_path)

        #convert numpy to tensor
        tensor_x = torch.from_numpy(arr).float()
        tensor_y = torch.from_numpy(lbl).long()
        ds = TensorDataset(tensor_x, tensor_y) #samples and labels 

        shuffle = True if split == "train" else False
        
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        datasets[split] = loader # mini batches
        logger.info(f"Loaded {len(ds)} samples for {split}")

    """
    3. create model
    """
    model = LandmarkNet(
        input_size=63,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
    ).to(device)

    """define loss function"""
    criterion = nn.CrossEntropyLoss()
    """define optimizer"""
    optimizer = Adam(model.parameters(), lr=lr)

    """ 
    4. Traninig pipeline
    """
    train_acc_history: list[float] = []
    val_acc_history: list[float] = []

    # Early stopping setup 
    patience = 5  # stop if val_acc doesn't improve for 5 epochs
    best_val_acc = -1.0
    patience_counter = 0

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        # training phase
        model.train()
        correct = 0
        total = 0
        for xb, yb in datasets["train"]: # mini batch
            xb = xb.to(device)
            yb = yb.to(device)

            #forward pass
            logits = model(xb)
            #loos calculation
            loss = criterion(logits, yb)
            #clear gradience
            optimizer.zero_grad()
            #backward pass
            loss.backward()
            # update parameters
            optimizer.step()

            preds = torch.argmax(logits, dim=1) # finde the higherst value in the logit for each hand of batch
            correct += (preds == yb).sum().item() #compares the guess with the answer key(yb), .sum = sums all the count(true=1,flase =0), .iteam = turns the tensor into regular python interger
            total += yb.size(0) # no of item in the current batch
        train_acc = correct / total if total > 0 else 0.0 #total correct/total = accuracy
        train_acc_history.append(train_acc)

        # validation phase
        model.eval()# set to validation mode
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

        # Early stopping check 
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
    training_time = time.time() - train_start # just loging start and end time

    """
    5. test evaluation
    """
    model.eval() # set to evaluation mode
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
            batch_size = xb.size(0)
            test_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = torch.argmax(logits, dim=1)
            #need cpu for accuracy (for eg for confusion matrix)
            all_preds.extend(preds.cpu().tolist()) #. extend: wraps all batched into single one long list
            all_labels.extend(yb.cpu().tolist())
    test_loss = test_loss / total_samples if total_samples > 0 else 0.0 # average loss per sample

    # calculating accuracy and other metrics
    test_accuracy = calculate_accuracy(all_labels, all_preds)
    metrics_dict = calculate_metrics(all_labels, all_preds, class_names)

    #saving the each trained model with the date it was trained
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"model_{ts}.pth"
    torch.save(model.state_dict(), model_path) # only saving weights 

    # This is just saving the important info and context
    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "landmarks",
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
            "hidden_layers": hidden_layers,
        },
    }
    metrics_path = models_dir / f"metrics_{ts}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # symlink update (get the freshest version from saved models for the real time inference)
    symlink_path = models_dir / "model_latest.pth"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    symlink_path.symlink_to(model_path.name)
    # log all other detailes
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
