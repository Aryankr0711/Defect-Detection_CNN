import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # Binary classification: 0 (Non-defective) and 1 (Defective)
DATASET_PATH = "dataset"  # 🔴 folder with class subfolders (0/ and 1/)
MODEL_SAVE_PATH = "cnn_pipeline_model.pth"
# 🔴 Set IMAGE_DIR if using CSV-based loading (path to folder containing images)
IMAGE_DIR = None  # Example: r"C:\Users\maila\Desktop\Defect_Detection\Normalised_Image_256"
CSV_PATH = "train_clean.csv"  # CSV file with ID and label columns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
# PREPROCESSING TRANSFORMS
# =========================
# Preprocessing pipeline: Resize + Normalize (equivalent to TensorFlow's Resizing + Rescaling)
preprocessing_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),  # Converts to [0, 1] range (equivalent to Rescaling 1./255)
])

# =========================
# DATASET CLASSES
# =========================
class ImageFolderDataset(Dataset):
    """Dataset for loading images from folder structure (class_0/, class_1/, etc.)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from subdirectories
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                class_label = int(class_name)  # Assuming folder names are "0", "1", etc.
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class CSVImageDataset(Dataset):
    """Dataset for loading images from CSV file (ID, label) and image directory"""
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path, dtype={"ID": str})
        self.image_dir = image_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["ID"]
        label = int(self.df.iloc[idx]["label"])
        
        # Try to find image with different extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate_path = os.path.join(self.image_dir, img_id + ext)
            if os.path.exists(candidate_path):
                img_path = candidate_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found for ID: {img_id}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# =========================
# CNN MODEL WITH PREPROCESSING PIPELINE
# =========================
class PreprocessingLayer(nn.Module):
    """Preprocessing layer that can be part of the model"""
    def __init__(self):
        super().__init__()
        # In PyTorch, we'll apply resize and normalization in forward pass
        # This is a wrapper to include preprocessing in the model pipeline
        self.resize = transforms.Resize(IMG_SIZE)
        self.to_tensor = transforms.ToTensor()
    
    def forward(self, x):
        # x is expected to be a PIL Image or batch of PIL Images
        if isinstance(x, Image.Image):
            x = self.resize(x)
            x = self.to_tensor(x)
            x = x.unsqueeze(0)  # Add batch dimension
        return x

class CNNModel(nn.Module):
    """CNN Model with preprocessing included in the pipeline"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # CNN layers (matching TensorFlow architecture)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size after conv layers
        # For 224x224 input: 224 -> 112 -> 56 -> 28 after 3 maxpool layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x should already be a tensor in [0, 1] range
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# DATA LOADING
# =========================
# Try to load from folder structure first, then fall back to CSV
USE_CSV = False

print(f"\n📂 Loading dataset...")
if os.path.exists(DATASET_PATH):
    print(f"  Found folder structure: {DATASET_PATH}")
    full_dataset = ImageFolderDataset(DATASET_PATH, transform=preprocessing_transform)
    print(f"✅ Loaded {len(full_dataset)} images from folder structure")
else:
    # Try CSV-based loading
    print(f"  Folder '{DATASET_PATH}' not found. Trying CSV-based loading...")
    if os.path.exists(CSV_PATH):
        # If IMAGE_DIR is not set, try to find it automatically
        if IMAGE_DIR is None:
            # Try common image directories in current folder
            possible_dirs = [
                "Combined_Resized_256",
                "Normalised_Image_256", 
                "Standardized_Image_256",
                "Renamed_Ok",
                "Renamed_Not_OK"
            ]
            
            for img_dir in possible_dirs:
                if os.path.exists(img_dir):
                    IMAGE_DIR = img_dir
                    break
        
        if IMAGE_DIR is None or not os.path.exists(IMAGE_DIR):
            print(f"\n❌ Error: Image directory not found!")
            print(f"   Please set IMAGE_DIR in export.py (around line 19)")
            print(f"   Example: IMAGE_DIR = r'C:\\path\\to\\your\\images'")
            print(f"\n   Or create a 'dataset' folder with this structure:")
            print(f"     dataset/")
            print(f"       ├── 0/  (Non-defective images)")
            print(f"       └── 1/  (Defective images)")
            exit(1)
        
        print(f"  Using CSV: {CSV_PATH}")
        print(f"  Using image directory: {IMAGE_DIR}")
        full_dataset = CSVImageDataset(CSV_PATH, IMAGE_DIR, transform=preprocessing_transform)
        USE_CSV = True
        print(f"✅ Loaded {len(full_dataset)} images from CSV")
    else:
        print(f"❌ Error: Neither dataset folder '{DATASET_PATH}' nor CSV file '{CSV_PATH}' found!")
        print("Please either:")
        print("  1. Create a dataset folder with structure: dataset/0/ and dataset/1/")
        print("  2. Or provide train_clean.csv and set IMAGE_DIR in the script")
        exit(1)

# Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                         generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"📊 Train samples: {train_size}, Validation samples: {val_size}")

# =========================
# CREATE MODEL
# =========================
model = CNNModel(num_classes=NUM_CLASSES).to(DEVICE)
print("\n📋 Model Architecture:")
print(model)

# =========================
# COMPILE MODEL (Loss & Optimizer)
# =========================
criterion = nn.CrossEntropyLoss()  # For multi-class (equivalent to sparse_categorical_crossentropy)
optimizer = optim.Adam(model.parameters())
print(f"\n✅ Model compiled with CrossEntropyLoss and Adam optimizer")

# =========================
# TRAIN MODEL
# =========================
print(f"\n🚀 Starting training for {EPOCHS} epochs...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for images, labels in train_pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * train_correct / train_total:.2f}%'
        })
    
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for images, labels in val_pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            val_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * val_correct / val_total:.2f}%'
            })
    
    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'model_config': {'num_classes': NUM_CLASSES, 'img_size': IMG_SIZE}
        }, MODEL_SAVE_PATH)
        print(f"  💾 Saved best model (Val Acc: {val_acc:.2f}%)")

print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")

# =========================
# EXPORT MODEL (PIPELINE)
# =========================
# Load the best model for export
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Save the complete model (including architecture)
torch.save(model, MODEL_SAVE_PATH.replace('.pth', '_complete.pth'))
print(f"\n✅ Model exported successfully at: {MODEL_SAVE_PATH}")
print(f"✅ Complete model (with architecture) saved at: {MODEL_SAVE_PATH.replace('.pth', '_complete.pth')}")

# =========================
# LOAD MODEL (FOR TESTING)
# =========================
print("\n🔄 Testing model loading...")
loaded_model = torch.load(MODEL_SAVE_PATH.replace('.pth', '_complete.pth'), map_location=DEVICE)
loaded_model.eval()
print("✅ Model loaded successfully")

# =========================
# SAMPLE INFERENCE FUNCTION
# =========================
def predict_single_image(image_path, model_path=None):
    """
    Predict a single image using the exported model.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model (default: uses the exported model)
    
    Returns:
        predicted_class: Class index (0 or 1)
        confidence: Confidence score
    """
    if model_path is None:
        model_path = MODEL_SAVE_PATH.replace('.pth', '_complete.pth')
    
    # Load model
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocessing_transform(img).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

# Example usage (uncomment to test)
if os.path.exists("2022.jpg"):
    class_id, conf = predict_single_image("test.jpg")
    print(f"\n📸 Test Prediction:")
    print(f"  Predicted class: {class_id} ({'Defective' if class_id == 1 else 'Non-defective'})")
    print(f"  Confidence: {conf:.2%}")

print("\n" + "="*60)
print("✅ Export pipeline completed successfully!")
print("="*60)
print(f"\n📝 Usage:")
print(f"  To use the exported model:")
print(f"    model = torch.load('{MODEL_SAVE_PATH.replace('.pth', '_complete.pth')}')")
print(f"    model.eval()")
print(f"\n  Or use the predict function:")
print(f"    class_id, confidence = predict_single_image('path/to/image.jpg')")
