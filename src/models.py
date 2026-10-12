import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


def scale_and_block_norm(blocks_tr, blocks_te):
    """
    Applies StandardScaler followed by L2 / sqrt(d) block normalization
    so high-dimensional transformer embeddings do not dominate lower-dimensional
    prosody, sentiment, and ASR confidence priors.
    """
    out_tr, out_te = [], []
    for b_tr, b_te in zip(blocks_tr, blocks_te):
        sc = StandardScaler()
        s_tr = sc.fit_transform(b_tr)
        s_te = sc.transform(b_te)
        scale = np.sqrt(b_tr.shape[1])
        out_tr.append(s_tr / scale)
        out_te.append(s_te / scale)
    return np.hstack(out_tr), np.hstack(out_te)


class GatedMultimodalNet(nn.Module):
    """
    Cross-modal gated fusion network combining acoustic and semantic branches.
    """
    def __init__(self, d_audio: int, d_text: int, n_classes: int = 4, hidden: int = 256, dropout: float = 0.35):
        super().__init__()
        self.aud_enc = nn.Sequential(
            nn.Linear(d_audio, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.txt_enc = nn.Sequential(
            nn.Linear(d_text, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x_aud: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        h_a = self.aud_enc(x_aud)
        h_t = self.txt_enc(x_txt)
        g = self.gate(torch.cat([h_a, h_t], dim=-1))
        fused = torch.cat([g * h_a, (1.0 - g) * h_t], dim=-1)
        return self.head(fused)


class BilinearGMUNet(nn.Module):
    """
    Bilinear Gated Multimodal Unit (GMU) with explicit multiplicative
    h_a * h_t interaction for capturing vocal tone vs. lexical sentiment contrast.
    """
    def __init__(self, d_audio: int, d_text: int, n_classes: int = 4, hidden: int = 256, dropout: float = 0.35):
        super().__init__()
        self.aud_proj = nn.Sequential(
            nn.Linear(d_audio, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(d_text, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.z_gate = nn.Linear(hidden * 2, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x_aud: torch.Tensor, x_txt: torch.Tensor) -> torch.Tensor:
        h_a = torch.tanh(self.aud_proj(x_aud))
        h_t = torch.tanh(self.txt_proj(x_txt))
        z = torch.sigmoid(self.z_gate(torch.cat([h_a, h_t], dim=-1)))
        gmu = z * h_a + (1.0 - z) * h_t
        bilinear = h_a * h_t
        diff = torch.abs(h_a - h_t)
        return self.classifier(torch.cat([gmu, bilinear, diff], dim=-1))
