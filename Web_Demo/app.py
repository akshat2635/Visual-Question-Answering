import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import os
import gdown
from VQA_model import *  

# Set Constants
DATA_FOLDER = "Web_Demo/data"
IMAGE_SIZE = (224, 224)
PREVIEW_SIZE = (300, 300)


file_id = "1Ec5tXFrPTNYD5GAn25ASj6nji867FPQA"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "Web_Demo/data/best_model.pth"

# Load Answer Vocabulary
@st.cache_resource
def load_answer_vocab():
    with open(os.path.join(DATA_FOLDER, "vocab.pkl"), "rb") as f:
        return pickle.load(f)

vocab = load_answer_vocab()
idx2answer = vocab["idx2answer"]

# Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False, fuzzy=True)
        print("Model Downloaded Successfully!")
    else:
        print("Model already exists, skipping download.")
    model = torch.load(os.path.join(DATA_FOLDER, "best_model.pth"), map_location="cpu",weights_only=False)
    model.eval()
    return model

model = load_model()

# Image transformation for model input
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🖼️ Visual Question Answering App")

# Image Upload
uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg", "webp"])

# Centered Image Preview
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    resized_image = image.resize(PREVIEW_SIZE)

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(resized_image, caption="Uploaded Image Preview", width=PREVIEW_SIZE[0])
    st.markdown("</div>", unsafe_allow_html=True)

# Question Input
question = st.text_input("❓ Enter a Question about the Image")

# Prediction
if uploaded_image and question:
    # Process Image
    image_tensor = transform(image).unsqueeze(0)

    # Encode Question
    encoded_question = torch.tensor(encode_question(question, vocab, max_q_len=20), dtype=torch.long).unsqueeze(0)

    # Get Prediction
    with torch.no_grad():
        output = model(image_tensor, encoded_question)
        predicted_idx = torch.argmax(output, dim=1).item()

    # Convert index to text answer
    predicted_text = idx2answer.get(predicted_idx, "<UNK>")

    # Display Answer with Larger Font
    st.markdown("""
        <div style='text-align: center; font-size: 24px; font-weight: bold;'>
            🏆 Predicted Answer: <br> <span style='color: #007BFF;'>{}</span>
        </div>
    """.format(predicted_text), unsafe_allow_html=True)
