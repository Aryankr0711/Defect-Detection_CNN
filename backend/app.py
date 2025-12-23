from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# -------------------------
# DEVICE
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MODEL ARCHITECTURE
# -------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load model
MODEL_PATH = "../cnn_pipeline_model.pth"  # Adjusted path
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = CNNModel(num_classes=2).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

IMG_SIZE = (224, 224)
preprocessing_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocessing_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    label_map = {
        1: "Non-defective",
        0: "Defective"
    }
    return label_map[pred_class], confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save uploaded file temporarily
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        try:
            label, conf = predict_image(temp_path)
            os.remove(temp_path)  # Clean up
            return jsonify({'prediction': label, 'confidence': f"{conf:.2%}"})
        except Exception as e:
            os.remove(temp_path)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
