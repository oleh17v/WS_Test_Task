import argparse
import torch
from PIL import Image
from transformers import BertTokenizerFast, BertForTokenClassification
from torchvision.transforms import v2
from torchvision import models
import json


class AnimalPipeline:
    def __init__(self, ner_model_path, image_model_path):
        # Load NER model and components
        self.ner_tokenizer = BertTokenizerFast.from_pretrained(ner_model_path)
        self.ner_model = BertForTokenClassification.from_pretrained(ner_model_path)
        with open(f"{ner_model_path}/label_map.json") as f:
            label_data = json.load(f)
            self.id2label = label_data["id2label"]

        # Load Image Classifier
        checkpoint = torch.load(image_model_path, map_location="cpu")
        self.classes = checkpoint['classes']
        self.image_model = models.resnet18(pretrained=False)
        self.image_model.fc = torch.nn.Linear(self.image_model.fc.in_features, len(self.classes))
        self.image_model.load_state_dict(checkpoint['model_state'])

        # Setup transforms and device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = self.image_model.to(self.device).eval()
        self.ner_model = self.ner_model.to(self.device).eval()
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def extract_animals(self, text):
        words = text.split()
        encoding = self.ner_tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.ner_model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding.word_ids()

        animals = set()
        current_animal = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            label = self.id2label[str(predictions[idx])]

            if label == "B-ANIMAL":
                if current_animal:
                    animals.add(" ".join(current_animal).lower())
                    current_animal = []
                current_animal.append(words[word_idx].lower())
            elif current_animal:
                animals.add(" ".join(current_animal).lower())
                current_animal = []

        return animals

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.image_model(tensor)

        return self.classes[output.argmax().item()].lower()

    def __call__(self, text, image_path):
        text_animals = self.extract_animals(text)
        if not text_animals:
            return False  # No animals mentioned in text

        image_animal = self.predict_image(image_path)
        return any(animal in image_animal for animal in text_animals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animal Verification Pipeline')
    parser.add_argument('--text', type=str, required=True, help='Input text description')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--ner_model', type=str, default='ner_model', help='Path to NER model')
    parser.add_argument('--image_model', type=str, default='models/image_classifier.pth',
                        help='Path to image classifier model')
    args = parser.parse_args()

    pipeline = AnimalPipeline(args.ner_model, args.image_model)
    result = pipeline(args.text, args.image)
    print(f"Verification Result: {result}")