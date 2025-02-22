import argparse
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification
from pathlib import Path


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()
        word_labels = self.labels[idx].split()

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(
                    self.label2id.get(word_labels[word_idx] if word_idx < len(word_labels) else self.label2id["O"],
                                      self.label2id["O"]))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


def main(args):
    # Load and prepare data
    df = pd.read_csv(args.data_path)
    label_list = ["O", "B-ANIMAL"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # Split into train and test sets (80% train, 20% test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["labels"].tolist(),
        test_size=0.2,  # 20% for testing
        random_state=42  # For reproducibility
    )

    # Initialize model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # Create datasets and dataloaders
    train_dataset = NERDataset(train_texts, train_labels, tokenizer, label2id)
    test_dataset = NERDataset(test_texts, test_labels, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}")

    # Save model and artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NER model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--output_dir', type=str, default='ner_model', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    args = parser.parse_args()
    main(args)