import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from timm import create_model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import altair as alt

# Ensure device is set to CPU
device = torch.device("cpu")
    
# Define number of classes and ensemble weights
num_classes = 30
weights = [0.3, 0.3, 0.4]  # [ViT weight, CNN weight, CNN-ViT weight]
vit_weight, cnn_weight, cnn_vit_weight = weights

# --------------------
# Label mapping
# --------------------

# Paths to the CSV and image directories
test_csv_path = 'book30-listing-test.csv'
# Read the CSV file
test_data = pd.read_csv(test_csv_path, encoding="ISO-8859-1")
test_labels = test_data.iloc[:, 6].tolist()

# Get the unique labels and create a mapping
label_names = [str(label) for label in sorted(np.unique(test_labels))]
label_to_index = {name: index for index, name in enumerate(label_names)}
# Encode labels as a numpy array
test_labels_encoded = np.array([label_to_index[label] for label in test_labels])

# --------------------
# Model Definitions
# --------------------

class EfficientNetLite0(nn.Module):
    def __init__(self, num_classes, feature_only=True, dropout_rate=0.5):
        super(EfficientNetLite0, self).__init__()

        # Load the pre-trained EfficientNet-Lite0 model
        self.model = create_model('tf_efficientnet_lite0', pretrained=False)
        
        # Extract features by removing the classification head (classifier)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])

        # Custom classifier for the desired number of classes
        self.classifier = nn.Linear(self.model.num_features, num_classes)
        
        # Global Average Pooling for consistent output size
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Apply dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Flag to toggle between feature extraction and classification
        self.feature_only = feature_only

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features

        if self.feature_only:
            return features  # Return raw features if feature_only is True

        features = self.global_pool(features)  # Apply global average pooling
        features = features.view(features.size(0), -1)  # Flatten features
        features = self.dropout(features)  # Apply dropout
        return self.classifier(features)  # Pass through the classifier

class HybridModel(nn.Module):
    def __init__(self, num_classes=30, use_vit=True, reduction_factor=1, dropout=0.1):
        super(HybridModel, self).__init__()
        self.use_vit = use_vit
        self.reduction_factor = reduction_factor

        # 🔹 Load EfficientNet-Lite0 from timm
        self.efficientnet = timm.create_model('tf_efficientnet_lite0', pretrained=False, num_classes=0, global_pool='')
        self.cnn_feature_dim = self.efficientnet.num_features
        self.cnn_avgpool = nn.AdaptiveAvgPool2d(1)

        if self.use_vit:
            # 🔹 Load ViT-Tiny from timm
            self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
            self.vit_feature_dim = self.vit.num_features

            # 🔹 Fusion Layer
            fusion_input_dim = self.cnn_feature_dim + self.vit_feature_dim
            self.fusion_fc = nn.Linear(fusion_input_dim, fusion_input_dim // self.reduction_factor)
        else:
            self.fusion_fc = None

        # 🔹 Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # 🔹 Final Classification Layer
        final_feature_dim = (self.fusion_fc.out_features if self.use_vit else self.cnn_feature_dim)
        self.classifier = nn.Linear(final_feature_dim, num_classes)

    def forward(self, x_cnn, x_vit=None):
        """
        Forward pass:
        - x_cnn: CNN input (image tensor for EfficientNet-Lite0)
        - x_vit: ViT input (optional, used only if use_vit=True)
        """
        # 🔹 CNN Pathway
        x_cnn = self.efficientnet.forward_features(x_cnn)
        x_cnn = self.cnn_avgpool(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)

        if self.use_vit and x_vit is not None:
            # 🔹 ViT Pathway
            x_vit = self.vit(x_vit)  # ViT model directly outputs features

            # 🔹 Feature Fusion
            x = torch.cat((x_cnn, x_vit), dim=1)
            x = self.fusion_fc(x)  # Reduce dynamically
        else:
            x = x_cnn  # CNN-only mode

        # 🔹 Dropout before classification
        x = self.dropout(x)
        
        # 🔹 Final Classification
        x = self.classifier(x)
        return x

# --------------------
# Load Models
# --------------------

# Initialize tokenizer for BERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Paths for the models
efficientnet_path = 'models/cnn.pth'
cnn_vit_path = 'models/cnn_vit.pth'
vit_path = 'models/vit.pth'
bert_path = 'models/bert'


# Load ViT model from the local path
vit_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
vit_state_dict = torch.load(vit_path, map_location=device)
vit_model.load_state_dict(vit_state_dict)

vit_model = vit_model.to(device)
vit_model.eval()


# Load CNN model from the local path
cnn_model = EfficientNetLite0(num_classes=30, feature_only=False)  # Ensure num_classes matches training
cnn_state_dict = torch.load(efficientnet_path, map_location=device)
cnn_model.load_state_dict(cnn_state_dict)

cnn_model = cnn_model.to(device)
cnn_model.eval()

# Load CNN-ViT model from the local path
cnn_vit_model = HybridModel(num_classes=num_classes, use_vit=True, reduction_factor=1)
cnn_vit_state_dict = torch.load(cnn_vit_path, map_location=device)
cnn_vit_model.load_state_dict(cnn_vit_state_dict)

cnn_vit_model.to(device)
cnn_vit_model.eval()

# Load BERT model from the local path
text_model = DistilBertForSequenceClassification.from_pretrained(bert_path, num_labels=30)
text_model = text_model.to(device)
text_model.eval()


# --------------------
# Prediction Functions
# --------------------

def cnn_predict(inputs, model):
    """
    Predict function for CNN model (EfficientNet-Lite0).
    Uses only the 'cnn' key.
    """
    images = inputs["cnn"]  # Extract RGB image for CNN

    with torch.no_grad():
        logits = model(images)  # Forward pass
        probabilities = F.softmax(logits, dim=1).cpu().numpy()

    return probabilities

def vit_predict(inputs, model):
    """
    Predict function for ViT model (ViT-Tiny).
    Uses only the 'vit' key.
    """
    images = inputs["vit"]  # Extract RGB image for ViT

    with torch.no_grad():
        logits = model(images)  # Forward pass
        probabilities = F.softmax(logits, dim=1).cpu().numpy()

    return probabilities

def cnn_vit_predict(inputs, model):
    """
    Predict function for Hybrid CNN-ViT model.
    Uses 'cnn_vit' key (Stacked CS1 + CS2).
    """
    images = inputs["cnn_vit"]  # Extract RGB + HSV stacked input

    # 🔹 Split into two separate inputs for CNN and ViT parts
    x_cnn = images[:, :3, :, :]  # First 3 channels -> RGB for CNN
    x_vit = images[:, 3:, :, :]  # Next 3 channels -> RGB for ViT

    with torch.no_grad():
        logits = model(x_cnn, x_vit)  # Forward pass through CNN-ViT model
        probabilities = F.softmax(logits, dim=1).cpu().numpy()  # Convert logits to probabilities

    return probabilities

def text_predict(input_ids, attention_mask, model):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # Forward pass to get logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()  # Apply softmax to logits
    return probabilities

# --------------------
# Classification Function
# --------------------
def classify_multimodal(image, text, tokenizer, text_model, vit_model, cnn_model, cnn_vit_model):
    # ---------- Preprocess Image ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    rgb_tensor = transform(image).unsqueeze(0).to(device)

    # Input formats for each model
    inputs = {
        "cnn": rgb_tensor,  # for the CNN model
        "vit": rgb_tensor,  # for the ViT model
        "cnn_vit": torch.cat([rgb_tensor, rgb_tensor], dim=1)  # if using RGB+RGB
    }
    
    encoded = tokenizer.encode_plus(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    
    with torch.no_grad():
        # ---------- Image Model Predictions ----------

        # Forward pass through image models
        vit_probs = vit_predict(inputs, vit_model)      # ViT uses XYZ
        cnn_probs = cnn_predict(inputs, cnn_model)      # CNN uses RGB
        cnn_vit_probs = cnn_vit_predict(inputs, cnn_vit_model)  # CNN-ViT uses RGB+HSV

        # 🔹 Move images to the correct device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        vit_probs = torch.tensor(vit_probs, dtype=torch.float32).to(device)
        cnn_probs = torch.tensor(cnn_probs, dtype=torch.float32).to(device)
        cnn_vit_probs = torch.tensor(cnn_vit_probs, dtype=torch.float32).to(device)

        # Combine image probs using weighted average
        image_probs = ((vit_probs + cnn_probs + cnn_vit_probs)/3).squeeze(0)  # shape: (num_classes,)

        # ---------- Text Model Prediction ----------

        # Forward pass through the text classifier
        text_probs_np = text_predict(encoded["input_ids"], encoded["attention_mask"], text_model)
        # Convert NumPy array to torch tensor
        text_probs = torch.tensor(text_probs_np, dtype=torch.float32, device=device)

        # ---------- Product-Max Fusion ----------
        # Multiply text_probs by the max of the image_probs (heuristic)
        fused_probs = text_probs * image_probs  # shape: (num_classes,)

        # ---------- Top-3 Predictions ----------
        top1 = fused_probs.argmax(dim=1)    #index in form of  tensor([28, 27, 21, 10], device='cpu')
        top3_indices = fused_probs.topk(3, dim=1).indices #index in form of tensor([28,20,27], [27, 1, 4], [21, 25, 5], [10, 0, 14], device='cpu')
        top3_confidence = fused_probs.topk(3, dim=1).values

        # Prepare the result by mapping each index to its confidence score
        top3_mapping = []
        for indices, confidences in zip(top3_indices, top3_confidence):
            mapping = list(zip(indices.tolist(), confidences.tolist()))  # Create pairs of (index, confidence score)
            top3_mapping.extend(mapping)  # Extend the list with the pairs

        
        # Create an index-to-label mapping from label_to_index mapping
        index_to_label = {index: label for label, index in label_to_index.items()}

        # Convert top3_mapping to a list of dictionaries for clearer presentation
        # Map the indices to their labels and prepare the list of dictionaries
        top3_with_labels = [
            {"Index": idx, "Label": index_to_label.get(idx, "Unknown"), "Confidence": conf}
            for idx, conf in top3_mapping
        ]



    return top3_with_labels


# --------------------
# Streamlit App
# --------------------
import streamlit as st
from PIL import Image

# --------------------
# Streamlit App
# --------------------
import streamlit as st
from PIL import Image

# Set page configuration to use wide layout
st.set_page_config(page_title="Multimodal Book Classification", layout="wide")

# Custom style to make the content wider while keeping it centered
st.markdown("""
    <style>
    .main {
        max-width: 1200px;  /* Increase max-width for wider content */
        margin: 0 auto;  /* Center the content */
    }
    
    /* Style the title */
    .stTitle {
        font-size: 36px;
        text-align: center;
        color: #4a4a4a;
    }

    /* Style description */
    .stMarkdown {
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }

    /* Style the button */
    .stButton > button {
        width: 100%;
        font-size: 20px;
        padding: 15px;
        background-color: #00bfae;
        color: white;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background-color: #009f93;
    }

    /* Style input fields */
    .stTextArea, .stFileUploader {
        width: 100%;
        font-size: 16px;
        padding: 12px;
        margin-bottom: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
    }

    .stTextArea textarea {
        font-size: 16px;
    }

    .stFileUploader input {
        font-size: 16px;
    }

    /* Style progress bar */
    .stProgress {
        margin-top: 10px;
    }

    .stSubheader {
        font-size: 24px;
        color: #333;
    }

    /* Style to display inputs side by side */
    .stColumns {
        display: flex;
        justify-content: space-between;
    }

    .prediction-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    .prediction-label {
        font-weight: bold;
    }

    /* Style the sidebar */
    .sidebar {
        width: 25%;  /* Set the sidebar width */
        padding: 20px;
        background-color: #f7f7f7;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for description
with st.sidebar:
    st.markdown("<h2>📚 BookFusion Classification App</h2>", unsafe_allow_html=True)
    st.markdown("""
    BookFusion allows you to classify books based on their title and cover image. 
    Simply enter the book's title and upload the cover image, and the app will predict the genre of the book based on multimodal classification.
    
    Trained using the BookCover30 dataset, it can identify 30 book genre classes.
                


    Accuracy: 63.80%
                
    Model: Multimodal Ensemble of CNN and ViT
                
    Dataset: BookCover30
    """)

# Main content area with wider layout
st.markdown('<h1 class="stTitle">📚 BookFusion Classification App</h1>', unsafe_allow_html=True)

# Text and image input section side-by-side
col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area("Enter book title:", key="text_input")
with col2:
    uploaded_file = st.file_uploader("Upload a book cover image", type=["jpg", "jpeg", "png"], key="uploaded_file")

#Button logic
if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif not uploaded_file:
        st.warning("Please upload an image.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        # Resize image if needed (for example, set the width to 800 while maintaining aspect ratio)
        image = image.resize((800, int(image.height * 800 / image.width)))
        
        # Call the unified multimodal classification function (replace with your actual function)
        top3_with_labels = classify_multimodal(
            image, text_input, tokenizer, text_model, vit_model, cnn_model, cnn_vit_model
        )
        
        # Create two columns: left for the image, right for results.
        col_img, col_results = st.columns(2)
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col_results:
            # Display the top-1 result prominently
            top1_label = top3_with_labels[0]['Label']
            top1_conf = top3_with_labels[0]['Confidence']
            st.markdown(f"### 🏆 Classification: **{top1_label}** — {top1_conf * 100:.2f}%")

            
            st.subheader("🔍 Top-3 Predictions")

            df = pd.DataFrame({
                'Label': [d['Label'] for d in top3_with_labels[:3]],
                'Confidence (%)': [d['Confidence'] * 100 for d in top3_with_labels[:3]]
            })

            # Create a horizontal bar chart using Altair
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Confidence (%)', title='Confidence (%)'),
                y=alt.Y('Label', sort='-x', title='Label')  # 'sort="-x"' orders bars descending by confidence
            ).properties(
                width=600,
                height=200
            )
            st.altair_chart(chart, use_container_width=True)





