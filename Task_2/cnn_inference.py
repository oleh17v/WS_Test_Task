import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import v2


def main(args):
    # Load model and classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Initialize model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint['classes']))
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device).eval()

    # Define transforms
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process image
    image = Image.open(args.image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(1).item()

    print(checkpoint['classes'][prediction])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image classification inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    main(args)