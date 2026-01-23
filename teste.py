import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class AnimalDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        self.labels_to_idx = {
            label: idx
            for idx, label in enumerate(self.data["label"].unique())
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]

        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, "dataset", img_path)

        label_str = self.data.iloc[idx]["label"]

        image = Image.open(img_path).convert("RGB")
        label = self.labels_to_idx[label_str]

        if self.transform:
            image = self.transform(image)
        
        return image, label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "dataset", "labels", "labels.csv")


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = AnimalDataset(csv_file=csv_path)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_indices, test_indices = random_split(
    range(len(dataset)),
    [train_size, test_size]
)

train_dataset = Subset(
    AnimalDataset(csv_file=csv_path, transform=train_transform),
    train_indices
)

test_dataset = Subset(
    AnimalDataset(csv_file=csv_path, transform=test_transform),
    test_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class AnimalModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.features = nn.Sequential(
           nn.Conv2d(input_shape, hidden_units, 3, padding=1),
           nn.BatchNorm2d(hidden_units),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units, hidden_units * 2, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 2),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units * 2, hidden_units * 4, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 4),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units * 4, hidden_units * 8, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 8),
           nn.ReLU(),

           nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
model = AnimalModel(3, 32, 10).to(device)

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, scheduler=None, device: torch.device = device):
  train_loss, train_acc = 0,0

  model.train()
  for batch, (X, y) in enumerate(data_loader):
    X,y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if scheduler:
        scheduler.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
  test_loss, test_acc = 0,0

  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X, y = X.to(device), y.to(device)

      test_pred = model(X)

      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")
    return test_loss, test_acc

loss_fn = nn.CrossEntropyLoss()
epochs = 500
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001) 
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.03, epochs=50, steps_per_epoch=len(train_loader), pct_start=0.3)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

for epoch in range(epochs):
   print(f"Epoch: {epoch}")
   train_step(model=model, data_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
   test_loss, test_acc = test_step(model, test_loader, loss_fn, accuracy_fn)

   if test_acc >= 99:
      break

from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "animal_recognition_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving to: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)