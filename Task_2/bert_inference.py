import argparse
import json
from pathlib import Path
import torch
from transformers import BertTokenizerFast, BertForTokenClassification


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and artifacts
    model = BertForTokenClassification.from_pretrained(args.model_dir).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)

    with open(Path(args.model_dir) / "label_map.json") as f:
        label_data = json.load(f)
        id2label = label_data["id2label"]

    # Process input text
    words = args.text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move to device and predict
    inputs = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # Process predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids()

    animals = []
    current_animal = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label = id2label[str(predictions[idx])]

        if label == "B-ANIMAL":
            if current_animal:
                animals.append(" ".join(current_animal))
                current_animal = []
            current_animal.append(words[word_idx])
        elif label == "O" and current_animal:
            animals.append(" ".join(current_animal))
            current_animal = []

    if current_animal:
        animals.append(" ".join(current_animal))

    print(animals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER Inference')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to trained model')
    parser.add_argument('--text', type=str, required=True, help='Input text for NER')
    args = parser.parse_args()
    main(args)