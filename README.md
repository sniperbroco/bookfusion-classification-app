# 📘 ReadMe: BookFusion – Ensemble of CNN, ViT and CNN-ViT Model

This document contains instructions for both training/testing the model (backend) and deploying it via a web application (frontend).

---

## 📑 Table of Contents

1. [🧠 Backend: Training & Testing via Jupyter](#1-backend-training--testing-via-jupyter)
   - [1.1 Prerequisites](#11-prerequisites)
   - [1.2 Environment Setup](#12-environment-setup)
   - [1.3 Running BookFusion-EfficientNetLite.ipynb](#13-running-bookfusion-efficientnetliteipynb)
   - [1.4 Running BookFusion-ViT-tiny.ipynb](#14-running-bookfusion-vit-tinyipynb)
   - [1.5 Running BookFusion-HybridModel.ipynb](#15-running-bookfusion-hybridmodelipynb)
   - [1.6 Running BookFusion-Ensemble-Image.ipynb](#16-running-bookfusion-ensemble-imageipynb)
   - [1.7 Running BookFusion-Multimodal.ipynb](#17-running-bookfusion-multimodalipynb)
   - [1.8 Output Files](#18-output-files)
2. [🌐 Frontend: Web Deployment via Streamlit](#2-frontend-web-deployment-via-streamlit)
   - [2.1 Environment Setup](#21-environment-setup)
   - [2.2 ▶️ Running the Web App](#22-running-the-web-app)

---

## 1. 🧠 Backend: Training & Testing via Jupyter

### 1.1 Prerequisites

#### Tools & Access
- Jupyter Notebook access  
- Python ≥ 3.8  
- GPU CUDA access (recommended)  
- A virtual environment (optional but cleaner)

#### Required Files
- `BookFusion-EfficientNetLite.ipynb` – CNN image model training & testing notebook  
- `BookFusion-ViT-tiny.ipynb` – ViT image model training & testing notebook  
- `BookFusion-HybridModel.ipynb` – CNN-ViT image model training & testing notebook  
- `BookFusion-DistilBERT.ipynb` – text model training & testing notebook  
- `BookFusion-Ensemble-Image.ipynb` – ensemble image model evaluation notebook  
- `BookFusion-Multimodal.ipynb` – multimodal model evaluation notebook  

#### Dataset Structure
bookcover30/
├── bookcover_images/

│ ├── 0001484524.jpg

│ ├── 006062213X.jpg

│ └── …

└── Task1/

├── book30-listing-test.csv

├── book30-listing-train.csv

├── bookcover30-labels-test.text

└── bookcover30-labels-train.text

- All images are in `224x224/`
- Use `.csv` files for full metadata; `.text` files contain labels only.
- Extract zip contents into a folder named `bookcover30`.

---

### 1.2 Environment Setup

Install all required Python packages:

```bash
pip install pandas numpy opencv-python-headless torch torchvision transformers Pillow tqdm matplotlib seaborn scikit-learn scikit-image timm nbimporter safetensors efficientnet-pytorch
```

### 1.3 Running `BookFusion-EfficientNetLite.ipynb`

1. Open the notebook in your IDE.
2. Run all cells:
   - Data loading  
   - Image preprocessing (augmentation)  
   - Model initialization  
   - Training loop  
   - Evaluation  
   - Model saving  
   - Result plotting  

---

### 1.4 Running `BookFusion-ViT-tiny.ipynb`

- Same steps as 1.3.  
- Make sure to run each cell in order.

---

### 1.5 Running `BookFusion-HybridModel.ipynb`

- Follow the same process:
  - Data loading → Preprocessing → Initialization → Training → Evaluation → Saving → Plotting

---

### 1.6 Running `BookFusion-Ensemble-Image.ipynb`

- Same steps:
  - Load predictions from individual models  
  - Ensemble logic  
  - Evaluate and compare

---

### 1.7 Running `BookFusion-Multimodal.ipynb`

1. Open the notebook.
2. Run all cells:
   - Load image and text data  
   - Initialize multimodal model  
   - Train and test  
   - Save outputs

---

### 1.8 Output Files

| File Type     | Directory | Description                |
|---------------|-----------|----------------------------|
| `.log`        | `logs/`   | Training/validation logs   |
| `.pth`        | `models/` | Best image model           |
| `.safetensors`| `models/` | Best text model            |

---

⚠️ **File Naming Consistency is Crucial**

Ensure consistency when saving and loading models:

```python
# Saving
torch.save(model.state_dict(), 'filename.pth')

# Loading
model.load_state_dict(torch.load('filename.pth'))
```

## 2. 🌐 Frontend: Web Deployment via Streamlit

This section guides you through deploying the trained BookFusion model via a web interface using **Streamlit**. The program is designed to run on a local machine.

---

### 2.1 Environment Setup

Make sure the following files are present in the same directory:

| File Name              | Purpose                                           |
|------------------------|---------------------------------------------------|
| `BookFusion.py`        | Main Streamlit deployment code                    |
| `/models/cnn.pth`      | Trained CNN image model (from backend)            |
| `/models/vit.pth`      | Trained ViT image model (from backend)            |
| `/models/cnn_vit.pth`  | Trained CNN-ViT image model (from backend)        |
| `/models/distilbert/`  | Trained text model                                |

Install the required dependencies:

```bash
pip install streamlit pandas numpy torch torchvision timm transformers pillow altair
```

### 2.2 ▶️ Running the Web App

1. Open a terminal and navigate to your project folder:

```bash
cd path/to/your/project/directory
```
💡 This makes sure you are in the correct directory where the files are located.

---

## 🚀 Launch the App

```bash
streamlit run BookFusion.py
```
By default, the app will automatically open in your browser at:

```bash
http://localhost:8501
```
### 📋 Features of the App
Once launched, the Streamlit app allows users to:

📚 Upload a book cover image and title for genre classification

📊 View top-1 and top-3 class predictions, ranked by confidence

⚠️ Model File Consistency
Ensure the model names in the code match the actual files:

```bash
# Paths for the models
efficientnet_path = 'models/cnn.pth'
cnn_vit_path     = 'models/cnn_vit.pth'
vit_path         = 'models/vit.pth'
bert_path        = 'models/distilbert'
```

- These lines can be found in BookFusion.py (around lines 139–143).
- Any filename mismatch will result in a FileNotFoundError.
- You can rename the model files or update the code paths accordingly.

If you have a GPU with CUDA, replace line [15] with the following:

```bash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
