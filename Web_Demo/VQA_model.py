import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Freeze most layers except the last few
        for param in list(self.cnn.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, images):
        img_features = self.cnn(images)  # (B, 2048, H, W) e.g. (B, 2048, 7, 7)
        img_features = self.avgpool(img_features)  # (B, 2048, 4, 4)
        return img_features


class MultiLayerAttention(nn.Module):
    def __init__(self, img_feat_dim=2048, q_feat_dim=2048, hidden_dim=1024, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Project image features into hidden space
        self.img_proj = nn.Conv2d(img_feat_dim, hidden_dim, kernel_size=1)  # (B, hidden_dim, 4, 4)
        # Project question features into hidden space
        self.q_proj = nn.Linear(q_feat_dim, hidden_dim)
        
        # Create attention layers and corresponding normalization layers
        self.attn_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
    def forward(self, img_features, q_features):
        B, C, H, W = img_features.shape
        # Project image features and flatten spatially: (B, 16, hidden_dim)
        img_proj = self.img_proj(img_features).view(B, -1, H * W).transpose(1, 2)
        # Project question feature and expand to spatial positions: (B, 1, hidden_dim) -> (B, 16, hidden_dim)
        q_proj = self.q_proj(q_features).unsqueeze(1)
        
        # Initialize attended image representation with projected image features
        attended_img = img_proj  # (B, 16, hidden_dim)
        
        for i in range(self.num_layers):
            # Expand question features to match spatial dimension
            q_proj_expanded = q_proj.expand(-1, attended_img.shape[1], -1)  # (B, 16, hidden_dim)
            # Combine current attended image features with question features
            combined = attended_img + q_proj_expanded  # (B, 16, hidden_dim)
            # Apply layer normalization
            combined = self.norm_layers[i](combined)
            # Compute attention scores using a non-linearity then linear projection: (B, 16)
            attn_scores = self.attn_layers[i](torch.tanh(combined)).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, 16)
            # Collapse spatial dimension to get a summary vector: (B, hidden_dim)
            attended_vector = torch.bmm(attn_weights.unsqueeze(1), attended_img).squeeze(1)
            # For the next layer, broadcast the summary vector back to spatial positions
            attended_img = attended_vector.unsqueeze(1).expand(-1, img_proj.shape[1], -1)
        
        # Final attended vector is used as image representation (B, hidden_dim)
        return attended_vector, attn_weights  # returning the last layer's attention weights for visualization if needed


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=3,
                            bidirectional=True, dropout=0.4)
        self.fc_q = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, questions):
        # questions: (B, seq_len)
        q_embed = self.embedding(questions)  # (B, seq_len, embed_dim)
        _, (q_hidden, _) = self.lstm(q_embed)
        # Concatenate last hidden states from both directions: (B, 2*hidden_dim)
        q_hidden = torch.cat((q_hidden[-2], q_hidden[-1]), dim=1)
        return self.fc_q(q_hidden)  # (B, 2048)


class VQAModel(nn.Module):
    def __init__(self, vocab_size, ans_vocab_size, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.question_encoder = QuestionEncoder(vocab_size, embed_dim, hidden_dim)
        self.attention_module = MultiLayerAttention(img_feat_dim=2048, q_feat_dim=2048, hidden_dim=1024, num_layers=3)
        
        self.fc_combined = nn.Sequential(
            nn.Linear(1024 + 2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc_final = nn.Linear(1024, ans_vocab_size)
        
    def forward(self, images, questions):
        img_features = self.image_encoder(images)         # (B, 2048, 4, 4)
        q_features = self.question_encoder(questions)       # (B, 2048)
        attended_img, attn_weights = self.attention_module(img_features, q_features)  # (B, 1024)
        combined = torch.cat((attended_img, q_features), dim=1)  # (B, 1024 + 2048)
        combined = self.fc_combined(combined)              # (B, 1024)
        return self.fc_final(combined)                     # (B, ans_vocab_size)


def encode_question(question,vocab,max_q_len=20):
    tokens = word_tokenize(question.lower())
    encoded = [vocab['word2idx'].get(word, vocab['word2idx']['<UNK>']) for word in tokens]
    return encoded + [0] * (max_q_len - len(encoded)) 