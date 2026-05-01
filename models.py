import torch
import torch.nn as nn
import math

class PINNJoint(nn.Module):
    def __init__(self, in_dim, context_dim=128):
        super().__init__()
        self.M_CENTER = 9.0
        self.M_SCALE  = 4.0
        self.S_CENTER = -1.5
        self.S_SCALE  = 3.5
        self.SIGMA_MIN = 0.01
        self.SIGMA_MAX = 2.00

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.m_head = nn.Linear(128, 2)   # z_M, log_var_M
        self.s_head = nn.Linear(128, 2)   # z_S, log_var_S

    def _bounded_mean_and_logvar(self, head_out, center, scale):
        mu = center + scale * torch.tanh(head_out[:, 0])
        log_var = head_out[:, 1].clamp(
            2 * math.log(self.SIGMA_MIN),
            2 * math.log(self.SIGMA_MAX)
        )
        return mu, log_var

    def forward(self, x):
        h = self.encoder(x)
        mu_M, lv_M = self._bounded_mean_and_logvar(self.m_head(h), self.M_CENTER, self.M_SCALE)
        mu_S, lv_S = self._bounded_mean_and_logvar(self.s_head(h), self.S_CENTER, self.S_SCALE)
        return {
            "context": h,                # 128‑dim, directly usable by the new flow
            "mu_mass": mu_M,
            "logvar_mass": lv_M,
            "mu_sfr": mu_S,
            "logvar_sfr": lv_S,
        }

# ----------------------
# TransientHunter Models
# ----------------------

class LightCurveEncoder(nn.Module):
    """
    LSTM-based encoder for light curve sequences
    Handles variable-length sequences with masking
    """
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, mask):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        encoded = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return encoded

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.2):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.projector(x)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

class TransientClassifier(nn.Module):
    """Main contrastive learning model"""
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, projection_dim=64, num_classes=10, dropout=0.2, use_features=True, feature_dim=40):
        super().__init__()
        self.use_features = use_features
        self.encoder = LightCurveEncoder(input_dim, hidden_dim, num_layers, dropout, True)
        
        encoder_output_dim = self.encoder.output_dim
        
        if use_features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            combined_dim = encoder_output_dim + hidden_dim
        else:
            combined_dim = encoder_output_dim
            
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, encoder_output_dim),
            nn.BatchNorm1d(encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.projection_head = ProjectionHead(encoder_output_dim, hidden_dim, projection_dim, dropout)
        self.classification_head = ClassificationHead(encoder_output_dim, hidden_dim, num_classes, dropout)
        
    def forward(self, sequence, mask, features=None, return_projection=False):
        seq_repr = self.encoder(sequence, mask)
        
        if self.use_features and features is not None:
            feat_repr = self.feature_encoder(features)
            combined = torch.cat([seq_repr, feat_repr], dim=-1)
            representation = self.fusion(combined)
        else:
            representation = seq_repr
            
        logits = self.classification_head(representation)
        
        result = {'representation': representation, 'logits': logits}
        if return_projection:
            result['projection'] = self.projection_head(representation)
        return result

# Baseline Autoencoder models
class AutoencoderEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.output_dim = hidden_dim * self.num_directions
        
    def forward(self, x, mask):
        x = self.input_proj(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        return lstm_out, (hidden, cell)

class AutoencoderDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=5, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        
    def forward(self, encoded, hidden_state):
        lstm_out, _ = self.lstm(encoded, hidden_state)
        return self.output_proj(lstm_out)

class AutoencoderClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        
    def forward(self, x): return self.classifier(x)

class LightCurveAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, num_classes=10, dropout=0.1, use_features=True, feature_dim=40):
        super().__init__()
        self.use_features = use_features
        self.encoder = AutoencoderEncoder(input_dim, hidden_dim, num_layers, dropout, False)
        self.decoder = AutoencoderDecoder(hidden_dim, input_dim, num_layers, dropout)
        
        if use_features:
            self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout))
            combined_dim = hidden_dim + hidden_dim // 2
        else:
            combined_dim = hidden_dim
            
        self.classification_head = AutoencoderClassificationHead(combined_dim, hidden_dim, num_classes, dropout)
        
    def forward(self, sequence, mask, features=None, return_reconstruction=True):
        encoded, hidden_state = self.encoder(sequence, mask)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-10)
        
        if self.use_features and features is not None:
            feat_encoded = self.feature_encoder(features)
            combined = torch.cat([pooled, feat_encoded], dim=-1)
        else:
            combined = pooled
            
        logits = self.classification_head(combined)
        result = {'representation': pooled, 'logits': logits}
        
        if return_reconstruction:
            result['reconstruction'] = self.decoder(encoded, hidden_state)
        return result
