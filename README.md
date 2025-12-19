# Defect Detection using CNN

A deep learning project for binary classification of defective and non-defective images using Convolutional Neural Networks (CNN) with PyTorch.

## 📋 Project Overview

This project implements a CNN-based binary classifier to detect defects in images. The model achieves high accuracy in distinguishing between defective (label=1) and non-defective (label=0) samples.

## 🏗️ Project Structure

```
Defect_Detection/
├── train.csv                      # Original training labels
├── train_clean.csv               # Cleaned training labels
├── train_1000_clean.csv          # Subset for testing
├── defect_cnn.pkl               # Trained model weights
├── train_CNN.ipynb             # Initial CNN training notebook
├── train_CNN_2_all.ipynb       # Complete CNN training pipeline
├── merge.ipynb                  # Data merging utilities
├── Remame_FIles.ipynb          # File renaming utilities
├── test.ipynb                   # Testing notebook
├── Ok/                          # Non-defective images
├── Not_OK/                      # Defective images
├── Renamed_Ok/                  # Renamed non-defective images
├── Renamed_Not_OK/              # Renamed defective images
├── Combined_Resized_256/        # Combined resized images (256x256)
├── Normalised_Image_256/        # Normalized images (256x256)
└── Standardized_Image_256/      # Standardized images (256x256)
```

## 🚀 Features

- **Binary Classification**: Defective vs Non-defective image classification
- **CNN Architecture**: Custom SimpleCNN with 3 convolutional layers
- **Data Preprocessing**: Image resizing, normalization, and standardization
- **High Accuracy**: Achieves >99% validation accuracy
- **PyTorch Implementation**: Modern deep learning framework
- **Data Augmentation**: Image transformations for better generalization

## 🛠️ Requirements

```bash
torch
torchvision
pandas
numpy
PIL (Pillow)
tqdm
matplotlib
```

## 📊 Dataset

- **Total Images**: 5,701 images
- **Classes**: 
  - Class 0 (Non-defective): ~4,700 images
  - Class 1 (Defective): ~1,000 images
- **Image Size**: 256x256 pixels
- **Format**: RGB images (JPG/PNG)

## 🧠 Model Architecture

### SimpleCNN
```python
- Conv2d(3, 16, 3) + ReLU + MaxPool2d(2)
- Conv2d(16, 32, 3) + ReLU + MaxPool2d(2)  
- Conv2d(32, 64, 3) + ReLU + MaxPool2d(2)
- Flatten + Linear(64*32*32, 128) + ReLU
- Linear(128, 1) # Binary output
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Aryankr0711/Defect-Detection_CNN.git
cd Defect-Detection_CNN
```

2. Install dependencies:
```bash
pip install torch torchvision pandas numpy pillow tqdm matplotlib
```

## 💻 Usage

### Training the Model

1. **Data Preparation**:
   - Run `merge.ipynb` to combine datasets
   - Use `Remame_FIles.ipynb` for file organization
   - Images are automatically resized to 256x256

2. **Training**:
   ```python
   # Open train_CNN_2_all.ipynb
   # Update image paths in the notebook
   # Run all cells to train the model
   ```

3. **Key Training Parameters**:
   - Batch Size: 32
   - Learning Rate: Default Adam optimizer
   - Epochs: 20
   - Train/Val Split: 80/20

### Model Performance

- **Training Accuracy**: >99%
- **Validation Accuracy**: >99%
- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam

### Inference

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('defect_cnn.pkl'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Predict
image = Image.open('path/to/image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = torch.sigmoid(model(input_tensor))
    prediction = (output > 0.5).float()
```

## 📈 Results

The model demonstrates excellent performance:
- Rapid convergence within first 10 epochs
- Stable training with minimal overfitting
- High precision and recall for both classes
- Robust performance on validation set

## 🔄 Data Pipeline

1. **Raw Images** → Organized into Ok/Not_OK folders
2. **Preprocessing** → Resize to 256x256, normalize pixel values
3. **Dataset Creation** → Custom PyTorch Dataset with CSV labels
4. **Training** → 80/20 train/validation split
5. **Evaluation** → Binary classification metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Aryan Kumar**
- GitHub: [@Aryankr0711](https://github.com/Aryankr0711)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Contributors to the dataset
- Open source community for tools and libraries

---

⭐ **Star this repository if you find it helpful!**