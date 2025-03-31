# Visual Question Answering (VQA)

This repository implements a series of deep learning models for Visual Question Answering (VQA). The project explores different architectures—ranging from a baseline model to advanced multi-modal fusion techniques—to tackle the challenging task of answering free-form questions about images.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Installation](#installation)
- [Team Members](#team-members)
- [License](#license)

## Overview

Visual Question Answering (VQA) is a multi-modal task where an AI system must generate accurate answers to questions based on the contents of an image. Unlike traditional image captioning, VQA requires a focused understanding of specific objects, actions, and contextual details within an image. This project implements four distinct VQA architectures that progressively incorporate advanced techniques from both computer vision and natural language processing.

## Models

### 1. Baseline Model (ResNet50 + LSTM)
- **Image Encoder:**  
  Utilizes a pretrained ResNet50 with its classification head removed. An adaptive average pooling layer resizes the feature maps (e.g., to a 2×2 grid), and a series of fully connected layers reduces the dimensionality to a 1024-dimensional vector.
  
- **Question Encoder:**  
  Processes the question using an embedding layer followed by a 3-layer bidirectional LSTM. The final hidden states are concatenated and processed to produce a 1024-dimensional question feature.
  
- **Fusion & Prediction:**  
  The image and question features are concatenated (forming a 2048-dimensional vector) and passed through additional fully connected layers to predict the answer.

### 2. Multi-Layer Attention with LSTM
- **Image Encoder:**  
  Similar to the baseline but uses adaptive pooling to produce a 4×4 grid (16 spatial regions).
  
- **Question Encoder:**  
  Uses the same LSTM-based approach to encode the question.
  
- **Attention Module:**  
  A multi-layer attention mechanism (3 layers) is employed. At each layer, attention weights are computed over the 16 spatial locations, refining the image features based on the question context.
  
- **Fusion & Prediction:**  
  The attended image features are concatenated with the question features and processed through fully connected layers to generate the answer.

### 3. Multi-Layer Attention + BERT
- **Image Encoder:**  
  Uses ResNet50 with adaptive pooling to obtain spatial feature maps.
  
- **Question Encoder:**  
  Replaces the LSTM with a pretrained BERT model (e.g., `bert-base-uncased`). The [CLS] token is extracted and further processed to match the required dimensionality.
  
- **Attention Module:**  
  Adapts the multi-layer attention mechanism to fuse the refined BERT-based question embeddings with the visual features.
  
- **Fusion & Prediction:**  
  The fused features (from attended image features and BERT embeddings) are processed through fully connected layers for answer prediction.

### 4. RCNN + Multi-Head Attention + BERT
- **Image Encoder:**  
  Utilizes a Faster R-CNN backbone (ResNet50 FPN) to extract fine-grained, object-level features. These are pooled and reshaped to yield a tensor of shape (B, 16, 256).
  
- **Question Encoder:**  
  Processes questions using BERT, with the [CLS] token serving as the initial feature that is refined through additional processing.
  
- **Attention Module:**  
  A multi-head attention mechanism (with 8 heads) allows the model to attend to multiple image regions simultaneously. The image features are projected to a 1024-dimensional space, and the BERT features are similarly projected to facilitate the attention operation.
  
- **Fusion & Prediction:**  
  Projected features from both modalities are fused via element-wise multiplication and then passed through fully connected layers to predict the answer.

## Dataset

The models are trained and evaluated on standard VQA benchmark datasets. Please refer to the dataset documentation provided with the repository for instructions on downloading and preparing the data.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akshat2635/Visual-Question-Answering.git
   cd Visual-Question-Answering
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Team Members

- **Bhavyadeep Singh Hada**
- **Ujjwal Jain**
- **Dev Pandya**
- **Akshat Jain**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
