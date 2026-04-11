"""
PhishingTransformer -- character-level Transformer encoder for phishing URL detection.

Architecture:
  Input  : URL string, padded to MAX_LEN characters
  Embed  : char embedding (64-dim) + learned positional encoding
  Encoder: 2 x TransformerEncoderLayer (4 heads, FFN dim 256, dropout 0.1)
  Pool   : mean pooling over non-padding positions
  FC     : 64 -> 1  (sigmoid = P phishing)

Usage
-----
Training (once):
    python train_model.py

Inference (per request):
    from services.ml_model import load_model, predict_risk
    model = load_model()
    score = predict_risk("evil-login.verify-paypal.tk", model)
"""

import os
import torch
import torch.nn as nn

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights.pth")

# -- Character vocabulary -------------------------------------------------
_VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
)
_CHAR2IDX      = {c: i + 1 for i, c in enumerate(_VOCAB)}  # 1-indexed; 0 = PAD
_VOCAB_SIZE    = len(_VOCAB) + 1        # known chars + PAD
_UNK_IDX       = _VOCAB_SIZE            # out-of-vocab
_VOCAB_TOTAL   = _VOCAB_SIZE + 1        # PAD + known + UNK

MAX_LEN  = 200
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
FFN_DIM  = 256


# -- Encoding helper -------------------------------------------------------

def encode_url(url: str) -> torch.Tensor:
    """
    Convert a URL / domain string to a padded integer tensor of shape (MAX_LEN,).
    """
    url = url.lower()[:MAX_LEN]
    indices = [_CHAR2IDX.get(c, _UNK_IDX) for c in url]
    indices += [0] * (MAX_LEN - len(indices))
    return torch.tensor(indices, dtype=torch.long)


# -- Model -----------------------------------------------------------------

class PhishingTransformer(nn.Module):
    """
    Lightweight Transformer encoder for character-level URL classification.

    forward(x) expects x of shape (batch, MAX_LEN) with integer char indices.
    Returns a (batch,) tensor of raw logits.
    """

    def __init__(self):
        super().__init__()
        self.char_embed = nn.Embedding(_VOCAB_TOTAL, D_MODEL, padding_idx=0)
        self.pos_embed  = nn.Embedding(MAX_LEN, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=0.1,
            batch_first=True,
            norm_first=True,         # Pre-LN: more stable training
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.dropout  = nn.Dropout(0.3)
        self.fc       = nn.Linear(D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, MAX_LEN)
        pad_mask = (x == 0)   # True at padding positions

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.char_embed(x) + self.pos_embed(positions)   # (batch, L, D)

        out = self.encoder(emb, src_key_padding_mask=pad_mask)  # (batch, L, D)

        # Mean pool over real (non-padding) tokens
        real = (~pad_mask).float().unsqueeze(-1)                # (batch, L, 1)
        pooled = (out * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)  # (batch, D)

        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)                      # (batch,)


# -- Training --------------------------------------------------------------

def train(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    """
    Train PhishingTransformer on the phishing URL dataset and save weights.pth.
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    from services.data_source import load_training_data

    print("=" * 55)
    print("  CertAgent -- Transformer Phishing Classifier")
    print("=" * 55)

    urls, labels = load_training_data()

    urls_tr, urls_val, y_tr, y_val = train_test_split(
        urls, labels, test_size=0.2, random_state=42, stratify=labels
    )

    class URLDataset(Dataset):
        def __init__(self, url_list, label_arr):
            self.urls   = url_list
            self.labels = torch.tensor(label_arr, dtype=torch.float32)
        def __len__(self): return len(self.urls)
        def __getitem__(self, idx):
            return encode_url(self.urls[idx]), self.labels[idx]

    train_loader = DataLoader(
        URLDataset(urls_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        URLDataset(urls_val, y_val), batch_size=batch_size, num_workers=0
    )

    model     = PhishingTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters : {total_params:,}")
    print(f"Training samples : {len(urls_tr):,}")
    print(f"Validation samples: {len(urls_val):,}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                probs = torch.sigmoid(model(xb))
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total   += len(yb)
        avg_loss = running_loss / len(train_loader)
        acc      = 100.0 * correct / total
        print(f"  Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={acc:.1f}%")

    path = os.path.abspath(_WEIGHTS_PATH)
    torch.save(model.state_dict(), path)
    print(f"\n[OK] Weights saved -> {path}")
    return model


# -- Inference -------------------------------------------------------------

def load_model():
    """Load trained model. Returns None if weights.pth does not exist."""
    path = os.path.abspath(_WEIGHTS_PATH)
    if not os.path.exists(path):
        print("[ml_model] weights.pth not found -- run `python train_model.py` first.")
        return None

    model = PhishingTransformer()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_risk(url_or_domain: str, model=None):
    """
    Return P(phishing) in [0.0, 1.0] for a URL or bare domain string.
    Returns None if the model has not been trained yet.
    """
    if model is None:
        model = load_model()
    if model is None:
        return None

    x = encode_url(url_or_domain).unsqueeze(0)
    with torch.no_grad():
        logit = model(x)
        prob  = torch.sigmoid(logit).item()
    return prob
