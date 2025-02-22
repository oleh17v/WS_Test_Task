import argparse
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from tqdm.auto import tqdm
from torchvision import models
from torchvision.transforms import v2

# Folder name translation dictionary
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}


def main(args):
    # Rename folders according to translation
    data_path = Path(args.data_dir)
    for folder in os.listdir(data_path):
        if folder in translate:
            old_path = data_path / folder
            new_path = data_path / translate[folder]
            os.rename(old_path, new_path)

    # Define transforms
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and split
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(
        dataset, [train_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, args.batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss, val_loss = 0, 0
        train_correct, val_correct = 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_data)
        val_loss /= len(test_loader)
        val_acc = val_correct / len(test_data)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save model
    model_path = Path(args.output_dir) / "model_0.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'classes': dataset.classes
    }, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an image classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for model')
    args = parser.parse_args()
    main(args)