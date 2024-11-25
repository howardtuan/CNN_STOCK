
import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import CNN5d

def load_model_for_prediction(model_path):
    model = CNN5d()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def predict_new_data(model, image_paths):
    predictions = []
    probabilities = []

    transform = transforms.Compose([
        transforms.Resize((32, 15)),
        transforms.ToTensor()
    ])

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('1')
            image_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            predictions.append(predicted_class)
            probabilities.append(probs.numpy().flatten())

    return predictions, probabilities

if __name__ == "__main__":
    model_path = './best_model.pth'
    model = load_model_for_prediction(model_path)

    image_paths = [
        './data/dataset(IMG&csv)/test_image1.png',
        './data/dataset(IMG&csv)/test_image2.png'
    ]

    predictions, probabilities = predict_new_data(model, image_paths)

    for img, pred, prob in zip(image_paths, predictions, probabilities):
        print(f"Image: {img}")
        print(f"Predicted Class: {pred}")
        print(f"Probabilities: {prob}")
