# CertAgent -- AI-Powered SSL Certificate Checker with Deep Learning Risk Scorer

A full-stack web security tool that combines **rule-based certificate analysis**, a **character-level Transformer neural network**, and **GPT-4o-mini** to assess the risk of any website.

---

## Project Structure

```
webCertificateChecker/
├── backend/
│   ├── app.py                        # Flask entry point
│   ├── train_model.py                # One-shot Transformer training script
│   ├── query_dataset.py              # Browse real URLs from the dataset
│   ├── requirements.txt
│   ├── weights.pth                   # Trained model weights (after training)
│   ├── phishing_cache.csv            # Downloaded dataset cache (after training)
│   ├── routes/
│   │   └── scan.py                   # POST /scan  POST /explain
│   └── services/
│       ├── cert_checker.py           # SSL certificate fetcher
│       ├── risk_analyzer.py          # Rule-based risk scoring
│       ├── llm_explainer.py          # GPT-4o-mini explanation (+ NN score injected)
│       ├── feature_extractor.py      # URL feature extraction (display/explainability)
│       ├── ml_model.py               # Transformer model definition + train/predict
│       └── data_source.py            # Dataset download, caching, URL parsing
└── frontend/
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── App.jsx
        ├── App.css
        └── components/
            ├── ScanForm.jsx
            ├── ResultCard.jsx         # Shows NN risk score bar
            └── AgentExplanation.jsx
```

---

## How It Works

```
User submits URL
      |
      v
 cert_checker          Fetch SSL certificate (Python ssl + socket)
      |
      v
 risk_analyzer         Rule-based checks: expiry / hostname / CA trust
      |                -> risk_level: Low / Medium / High
      v
 PhishingTransformer   Character-level Transformer scores the domain string
      |                -> nn_risk_score: 0-100%  (P phishing)
      v
 llm_explainer         GPT-4o-mini explains both scores in plain English
      |                (NN score is injected into the prompt)
      v
 Frontend              Displays cert details + risk badge + NN score bar + AI explanation
```

### Two Independent Risk Engines

| Engine | Input | What it measures |
|---|---|---|
| Rule-based | SSL certificate fields | Is the certificate itself valid? |
| Transformer NN | Domain name characters | Does the URL structure look like phishing? |

These complement each other. A site can have a valid certificate but a suspicious URL structure (common in phishing sites using Let's Encrypt), or a suspicious certificate on a legitimate domain.

---

## Dataset

**Malicious URL Detection Deep Learning Dataset**

| Property | Value |
|---|---|
| Source | `incertum/cyber-matrix-ai` (GitHub) |
| Size | **194,798 URLs** |
| Balance | 97,399 safe + 97,399 phishing (50/50) |
| Columns | `url`, `isMalicious` (0 = safe, 1 = phishing) |
| Origin | PhishTank, OpenPhish (phishing) + Alexa Top Sites (safe) |
| Local cache | `backend/phishing_cache.csv` (downloaded automatically on first train) |

Train / validation split: **80% / 20%** via `sklearn.train_test_split` with `stratify=labels` to preserve class balance in both splits.

- Training set : 155,838 URLs
- Validation set: 38,960 URLs

---

## Model Evolution -- Three Versions Compared

All three models were evaluated on the **same 38,960 validation URLs** (random_state=42, stratify=labels).

### Version History

| Version | Model | Architecture | Data | Val Accuracy |
|---|---|---|---|---|
| v1 | MLP | 7->16->8->1, hand-crafted features | Synthetic 10k | 91.0% (inflated -- synthetic data) |
| v2 | MLP | 12->32->16->1, 12 URL features | Real 194k | 65.2% |
| v3 | MLP | 12->32->16->1 + entropy/TLD | Real 194k | 65.2% |
| v4 | **TextCNN** | char-level Conv1D (k=3,4,5), 128 filters | Real 194k | **96.36%** |
| v5 | **Transformer** | char-level encoder, 2 layers, 4 heads | Real 194k | 90.08% |

---

## Full Benchmark: Rule-Based vs TextCNN vs Transformer

Evaluated on the same 38,960-URL validation set. Fair comparison: identical data split, identical threshold (>= 0.5 = phishing).

### Accuracy & AUC-ROC

| Engine | Accuracy | AUC-ROC |
|---|---|---|
| Rule-based (cert checks only) | 55.25% | -- |
| TextCNN (character-level) | **96.36%** | **0.9938** |
| Transformer (character-level) | 90.08% | 0.9647 |

> Rule-based accuracy was measured by treating "Low risk" as safe and "Medium/High" as phishing on the 38,960 URLs. It only considers certificate validity, not URL structure.

### Error Rates

| Engine | FPR (safe flagged as phishing) | FNR (phishing missed) |
|---|---|---|
| Rule-based | -- | -- |
| TextCNN | **4.02%** | **3.25%** |
| Transformer | 10.34% | 9.49% |

### Confusion Matrices

**TextCNN (96.36% accuracy)**
```
                  Pred Safe    Pred Phishing
  True Safe          18,696              784
  True Phishing         634           18,846
```

**Transformer (90.08% accuracy)**
```
                  Pred Safe    Pred Phishing
  True Safe          17,465            2,015
  True Phishing       1,848           17,632
```

### Classification Report -- TextCNN

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.9672 | 0.9598 | 0.9635 |
| Phishing | 0.9601 | 0.9675 | 0.9637 |
| **Macro avg** | **0.9637** | **0.9637** | **0.9636** |

### Classification Report -- Transformer

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.90 | 0.90 | 0.90 |
| Phishing | 0.90 | 0.91 | 0.90 |
| **Macro avg** | **0.90** | **0.90** | **0.90** |

---

## Why TextCNN Outperforms Transformer on URL Classification

URLs are short strings (average ~50 characters) where **local n-gram patterns** are the primary signal:

- Suspicious substrings: `login`, `verify`, `secure`, `paypal`, `account`, `.php?`
- Suspicious TLDs: `.tk`, `.ml`, `.ga`, `.cf`
- Structure patterns: IP addresses as host, excessive hyphens/dots, long random strings

TextCNN's parallel Conv1D filters (kernel sizes 3, 4, 5) detect these patterns at **any position** efficiently. Transformers excel at **long-range contextual dependencies** (e.g., understanding sentence meaning in NLP), but URL classification doesn't need that -- it is fundamentally an n-gram pattern matching problem. The Transformer also needs more training epochs to converge on short inputs.

| Property | TextCNN | Transformer |
|---|---|---|
| Parameters | ~123K | ~117K |
| Inductive bias | Local n-gram patterns | Global context |
| Suited for | Short pattern-matching strings | Long documents, cross-token meaning |
| Training speed (CPU) | Fast (~5 min / 10 epochs) | Slower (~8 min / 10 epochs) |
| Val accuracy @ epoch 10 | 94.6% | 90.1% |
| Val accuracy @ convergence | **96.36%** (epoch 60) | ~91% (estimated) |

**Conclusion:** For character-level URL phishing detection, TextCNN is the better architecture. The current production model uses the Transformer (v5); swap `ml_model.py` to the TextCNN class and retrain to use v4.

---

## Deep Learning Model Architectures

### TextCNN (v4 -- Best Accuracy: 96.36%)

```
Input : URL string (lowercased, padded to 200 chars)
        |
Embed : each char -> 16-dim vector   [vocab: a-z, 0-9, URL special chars]
        |
Conv1D: 3 parallel filters (kernel sizes 3, 4, 5) x 128 filters each
        |
MaxPool: global max over sequence length
        |
Concat: 384-dim  (128 x 3 kernels)
        |
Dropout(0.5)
        |
FC    : 384 -> 128 -> ReLU -> Dropout(0.3) -> 1
        |
Sigmoid -> P(phishing)
```

### Transformer (v5 -- Current Production Model: 90.08%)

```
Input : URL string (lowercased, padded to 200 chars)
        |
Embed : char embedding (64-dim) + learned positional encoding
        |
Encoder: 2 x TransformerEncoderLayer
         (4 heads, FFN dim 256, dropout 0.1, Pre-LN norm_first=True)
        |
Pool  : mean pooling over non-padding positions
        |
Dropout(0.3)
        |
FC    : 64 -> 1
        |
Sigmoid -> P(phishing)
```

### Training Configuration (both models)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Learning rate | 1e-3 |
| LR schedule | OneCycleLR |
| Epochs | 10 (Transformer) / 60 (TextCNN) |
| Batch size | 256 |
| Loss | BCEWithLogitsLoss |
| Gradient clipping | 1.0 |

---

## Setup & Running

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (once)

```bash
cd backend
python train_model.py
```

Expected output:
```
-- Transformer smoke-test --
  Tensor : shape=(200,), dtype=torch.int64
  Params : 116,737
  Untrained P(phishing) = 0.498  (random weights, expect ~0.5)

CertAgent -- Transformer Phishing Classifier
Model parameters : 116,737
Training samples : 155,838
Validation samples: 38,960

  Epoch  1/10  loss=0.5827  val_acc=79.2%
  Epoch  5/10  loss=0.2764  val_acc=88.7%
  Epoch 10/10  loss=0.2075  val_acc=90.1%

[OK] Weights saved -> backend/weights.pth
```

This downloads the dataset automatically (cached as `phishing_cache.csv` for future runs).

### 3. Set OpenAI API key (optional)

```bash
export OPENAI_API_KEY=sk-...          # Linux/Mac
$env:OPENAI_API_KEY="sk-..."          # Windows PowerShell
```

The app works without a key -- AI explanation is skipped.

### 4. Start backend

```bash
cd backend
python app.py
# Running on http://127.0.0.1:5000
```

### 5. Start frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
# Running on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## API Endpoints

### `POST /scan`
Returns certificate info + rule-based risk level + NN risk score immediately.

**Request**
```json
{ "url": "https://expired.badssl.com" }
```

**Response**
```json
{
  "domain": "expired.badssl.com",
  "status": "invalid",
  "risk_level": "High",
  "nn_risk_score": 95.7,
  "certificate": {
    "issuer": "...",
    "valid_from": "2015-04-09",
    "valid_to": "2015-04-12",
    "days_remaining": -3285,
    "hostname_match": true
  },
  "analysis": {
    "findings": ["Certificate expired 3285 days ago."],
    "summary": "Certificate has critical problems that pose a security risk."
  }
}
```

### `POST /explain`
Calls GPT-4o-mini with both rule-based findings and NN score injected into the prompt.

**Response**
```json
{
  "ai_explanation": "...",
  "recommendations": ["...", "..."],
  "nn_risk_score": 95.7
}
```

---

## Test Domains

### From the real dataset (resolvable)

| Domain | True Label | NN Score | Notes |
|---|---|---|---|
| `161.113.4.71` | Phishing | 98.0% | IP address -- strong phishing signal |
| `mahirmobilya.com` | Phishing | 71.2% | |
| `unlimitedtrekkingnepal.com` | Phishing | 75.5% | |
| `hbu.edu.cn` | Safe | 2.0% | Clean education domain |
| `tuniu.com` | Safe | 48.6% | |

### BadSSL test cases (cert edge cases)

| Domain | Rule Risk | NN Score | What to observe |
|---|---|---|---|
| `expired.badssl.com` | High | 95.7% | Both engines flag it |
| `wrong.host.badssl.com` | High | 96.7% | Both engines flag it |
| `self-signed.badssl.com` | Medium | 95.6% | Both engines flag it |
| `sha256.badssl.com` | **Low** | **97.4%** | Cert valid but URL structure suspicious |
| `badssl.com` | Low | 77.1% | |

`sha256.badssl.com` is the most interesting case: the certificate is perfectly valid (rule engine says Low Risk) but the NN scores it 97.4% because sub-domain naming with digits (`sha256`) resembles phishing patterns in the training data -- demonstrating why both engines are needed.

---

## Browsing the Dataset

```bash
cd backend
python query_dataset.py 10 phish    # 10 phishing URLs
python query_dataset.py 10 safe     # 10 safe URLs
python query_dataset.py 20          # 20 random mixed
```
