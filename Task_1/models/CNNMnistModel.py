import torch
from torch import nn, optim
from torchmetrics import Accuracy
from MnistClassifierInterface import MnistClassifierInterface


# b. Convolutional Neural Network (CNN)
class CNNMnistModel(nn.Module, MnistClassifierInterface):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = 5
        self.accuracy = Accuracy(task="multiclass", num_classes=10).to(self.device)

    def _build_model(self):
        # Define the CNN architecture directly
        model = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Classifier
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)
        )
        return model

    def train(self, train_loader, val_loader):
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_acc += self.accuracy(y_pred.argmax(dim=1), y).item()

            # Validation
            self.model.eval()
            val_loss, val_acc = 0.0, 0.0
            with torch.inference_mode():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    y_pred = self.model(X)
                    loss = self.criterion(y_pred, y)

                    val_loss += loss.item()
                    val_acc += self.accuracy(y_pred.argmax(dim=1), y).item()

            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}")
            print(f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2f}\n")

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.inference_mode():
            for X, _ in test_loader:
                X = X.to(self.device)
                y_pred = self.model(X)
                predictions.append(y_pred.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()
