import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Spatial Attention Module
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # 7x7 convolution to compute an attention map
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn

# ---------------------------
# Appearance Stream Module
# ---------------------------
class AppearanceStream(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=3, dim=128, heads=4):
        super(AppearanceStream, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        self.flatten = nn.Flatten(2)

        # Patch embedding
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_attn = SpatialAttention(dim)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer Encoder
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Final FC layer to project features
        self.fc = nn.Linear(dim, 128)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # -> [B, dim, H/P, W/P]
        x = self.spatial_attn(x)
        x = x.flatten(2).transpose(1, 2)  # -> [B, num_patches, dim]
        x = x + self.pos_embed

        # Transformer block
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(attn_out + x)
        x = self.norm2(self.mlp(x) + x)

        # Global average pooling
        x = x.mean(dim=1)  # -> [B, dim]
        return self.fc(x)


# ---------------------------
# Motion Stream Module
# ---------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        scores = self.fc(x).squeeze(-1)  # [batch, seq_len]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        weighted = x * weights  # [batch, seq_len, hidden_dim]
        return weighted.sum(dim=1)  # [batch, hidden_dim]

class MotionStream(nn.Module):
    def __init__(self):
        super(MotionStream, self).__init__()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.attn = TemporalAttention(hidden_dim=128)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, seq_len, 128]
        out = self.attn(lstm_out)   # apply temporal attention
        return out  # [batch, 128]


# ---------------------------
# Updated HTNet Model (Two-Stream)
# ---------------------------
class HTNet(nn.Module):
    def __init__(self, image_size, patch_size, dim, heads, num_hierarchies, block_repeats, num_classes):
        """
        Updated HTNet using a two-stream architecture:
            - AppearanceStream for spatial features.
            - MotionStream for temporal (motion) features.
        The outputs of both streams (each 128-dim) are concatenated and
        passed through fully connected layers to produce logits.
        """
        super(HTNet, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.heads = heads
        self.num_hierarchies = num_hierarchies
        self.block_repeats = block_repeats
        self.num_classes = num_classes
        
        self.appearance = AppearanceStream(img_size=28, patch_size=7, in_channels=3, dim=128, heads=4)
        self.motion = MotionStream()
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, appearance_input, motion_input=None, return_features=False):
        """
        Parameters:
            appearance_input: Tensor of shape (batch, 1, 224, 224)
                (preprocessed face images)
            motion_input: Tensor of shape (batch, sequence_length, 64)
                (per-frame features from a video sequence; if not provided, defaults to zeros)
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        feat_app = self.appearance(appearance_input)
        if motion_input is not None:
            feat_motion = self.motion(motion_input)
        else:
            feat_motion = torch.zeros(feat_app.size(0), 128, device=feat_app.device)
        
        # Concatenate along the feature dimension
        combined_features = torch.cat([feat_app, feat_motion], dim=1)
        if return_features:
            return combined_features
        logits = self.fc(combined_features)
        return logits

# ---------------------------
# Fusion Model (Unchanged, with Minor Naming Adjustments)
# ---------------------------
class Fusionmodel(nn.Module):
    def __init__(self):
        super(Fusionmodel, self).__init__()
        self.fc1 = nn.Linear(15, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        # Each part feature shape: [B, 3]
        # Concatenate along feature dimension, not batch
        fuse_five_features = torch.cat(
            [l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature], dim=1
        )  # → [B, 15]
        
        fuse_out = self.fc1(fuse_five_features)  # → [B, 3]
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)

        fuse_whole_five_parts = torch.cat([whole_feature, fuse_out], dim=1)  # [B, 3 + 3 = 6]
        fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
        fuse_whole_five_parts = self.d1(fuse_whole_five_parts)

        out = self.fc_2(fuse_whole_five_parts)  # → [B, num_classes]
        return out


# ---------------------------
# Quick Test (Optional)
# ---------------------------
if __name__ == "__main__":
    # Testing the updated HTNet with dummy data
    
    # Instantiate the model for a 3-class classification problem
    model = HTNet(num_classes=3)
    
    # Create a dummy appearance input (batch of 2 images, 1 channel, 224x224)
    dummy_appearance = torch.randn(2, 1, 224, 224)
    # Create a dummy motion input (batch of 2, sequence_length=16, feature_dim=64)
    dummy_motion = torch.randn(2, 16, 64)
    
    # Forward pass: using both streams
    outputs = model(dummy_appearance, dummy_motion)
    print("Output shape (two-stream):", outputs.shape)
    
    # Forward pass: using only the appearance stream (motion_input not provided)
    outputs = model(dummy_appearance)
    print("Output shape (appearance only):", outputs.shape)
