from sys import path
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models import CifarCNN
from Evaluationfunction import evaluate 
#preparing the device
training_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#H.Ps
L_R = 0.001
EPOCHS = 20
BATCH_SIZE = 64
TRAIN_SIZE = 45000
VAL_SIZE = 5000

# --- Data Preparation ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616)
)
])


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

# --- Load Datasets ---
full_train_dataset = datasets.CIFAR10(
    root="D:/datasets",
    train=True,
    download=True,
    transform=train_transform
)


test_dataset = datasets.CIFAR10(
    root="D:/datasets",
    train=False,
    download=True,
    transform=eval_transform
)
# --- Create DataLoaders ---
train_dataset, val_dataset = random_split(
full_train_dataset, [TRAIN_SIZE, VAL_SIZE],
generator=torch.Generator().manual_seed(42)
    )
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#model definition
model = CifarCNN().to(training_device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=L_R)

# --- Training Loop ---
metric = {
        "dataset": "CIFAR-10",
        "epochs": [],
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": L_R,
            "optimizer": "Adam",
            "epochs": EPOCHS,
        },
    }
for epoch in range(EPOCHS):
    model.train()  
    running_loss = 0.0
    for images, labels in tqdm(train_loader,desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(training_device), labels.to(training_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"  Epoch [{epoch + 1}/{EPOCHS}]  ─  Avg Loss: {avg_loss:.4f}")


    val_acc = evaluate(model, val_loader, training_device, label="Validation")
    test_acc = evaluate(model, test_loader, training_device, label="Test")
    metric["epochs"].append({
            "epoch": epoch + 1,
            "avg_loss": round(avg_loss, 4),
            "val_accuracy": round(val_acc, 2),
            "test_accuracy": round(test_acc, 2),
        })


     # --- Save Trained Model ---
    os.makedirs("./CHECKPOINT_DIR", exist_ok=True)
    torch.save(model.state_dict(), "./CHECKPOINT_DIR/model.pth")
    print(f"\nModel checkpoint saved to: ./CHECKPOINT_DIR/model.pth")

    # --- Save Training Metrics ---
    os.makedirs("./METRICS_DIR", exist_ok=True)
    with open("./METRICS_DIR/metrics.json", "w") as f:
        json.dump(metric, f, indent=2)
    print(f"Training metrics saved to: ./METRICS_DIR/metrics.json")


