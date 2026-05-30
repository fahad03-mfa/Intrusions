# 🛡️ IoT & 5G Intrusion Detection System — Deep Learning + Ensemble CNN

> **Cognitive Security Framework** | Deep Learning | Computer Vision | Flask | CICIDS & UNSW-NB15

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

---

## 🔍 What This Project Does

A **production-ready Intrusion Detection System (IDS)** for IoT and 5G networks that takes raw network traffic (CSV), converts it into visual image representations, and classifies it as **BENIGN or ATTACK** using a dual-CNN ensemble architecture.

The key novelty: instead of treating network features as tabular data, this system **transforms them into RGB and Grayscale images** and leverages convolutional neural networks — bringing computer vision techniques to the cybersecurity domain.

---

## 🚀 Key Highlights

| Feature | Detail |
|---|---|
| **Approach** | Feature-to-image transformation + CNN classification |
| **Datasets** | CICIDS 2017, CICIDS 2018, UNSW-NB15 |
| **Models** | Dual CNN (RGB + Grayscale) with weighted ensemble |
| **Class Imbalance** | SMOTE oversampling + random undersampling |
| **Deployment** | Flask web app — upload CSV, get instant prediction |
| **Domain** | IoT Security · 5G Networks · Network Forensics |

---

## 🧠 Architecture & Methodology

```
Raw Network Traffic (CSV)
         │
         ▼
┌──────────────────────┐
│  Data Preprocessing  │  MinMaxScaler + Label Encoding + Binary Classification
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌──────────┐
│  RGB   │   │Grayscale │   Feature → Image Conversion (64×64 px)
│ Image  │   │  Image   │   Cubic Interpolation for Grayscale
└───┬────┘   └────┬─────┘
    │              │
    ▼              ▼
┌────────┐   ┌──────────┐
│CNN RGB │   │CNN Gray  │   Separate trained models per modality
└───┬────┘   └────┬─────┘
    │              │
    └──────┬───────┘
           ▼
  ┌──────────────────┐
  │ Weighted Ensemble │   RGB weight: 0.98 | Gray weight: 0.94
  └────────┬─────────┘
           ▼
  BENIGN  /  ATTACK
```

### Why Feature-to-Image?
Traditional ML methods treat network flows as flat vectors. This approach **encodes relational structure** between features spatially, enabling CNNs to learn patterns across multiple features simultaneously — a technique gaining traction in network security research.

---

## 📂 Repository Structure

```
Intrusions/
├── app/
│   ├── preprocessing.py       # Feature scaling and cleaning
│   ├── image_converter.py     # RGB and grayscale image generation
│   └── main.py                # Flask routes and model inference
├── models/
│   ├── model_rgb.h5           # Trained CNN for RGB images
│   └── model_gray.h5          # Trained CNN for Grayscale images
├── templates/
│   └── index.html             # Web UI for CSV upload + prediction
├── static/images/             # Generated visualization outputs
├── output/                    # Prediction results
├── main.py                    # Flask application entry point
├── model_train.ipynb          # Full training pipeline notebook
├── scaler.joblib              # Saved MinMaxScaler
├── instance1.csv              # Sample test instance (CICIDS)
├── instance2.csv              # Sample test instance (UNSW-NB15)
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/fahad03-mfa/Intrusions.git
cd Intrusions

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Flask application
python main.py
```

Visit `http://127.0.0.1:5000` — upload a network traffic CSV and get a prediction instantly.

### Input Format
Upload a `.csv` file containing network flow features (compatible with CICIDS or UNSW-NB15 feature sets). Sample files `instance1.csv` and `instance2.csv` are included.

---

## 🔬 Technical Deep Dive

### 1. Feature-to-Image Conversion

**RGB Mode:** Each normalized feature value is mapped to a 24-bit RGB color using:
```python
rgb_value = int(value * 16777215)  # 2^24 - 1 colors
r = (rgb_value >> 16) & 0xFF
g = (rgb_value >> 8) & 0xFF
b = rgb_value & 0xFF
```
Features are distributed across a 64×64 pixel canvas.

**Grayscale Mode:** Features are arranged in a 2D grid, then upscaled to 64×64 using **cubic spline interpolation** (`RegularGridInterpolator`), preserving spatial relationships between features.

### 2. Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) to generate realistic synthetic minority class samples
- **Random undersampling** to trim dominant class
- Result: balanced training distribution, unbiased recall

### 3. Weighted Ensemble Strategy
```python
# RGB model has slightly higher weight due to richer color encoding
combined = (rgb_pred * 0.5098) + (gray_pred * 0.4902)
final_class = 1 if combined > 0.54 else 0  # ATTACK or BENIGN
```

---

## 📊 Results

| Metric | RGB Model | Grayscale Model | Ensemble |
|---|---|---|---|
| Accuracy | — | — | — |
| Precision | — | — | — |
| Recall | — | — | — |
| F1 Score | — | — | — |

> ⚠️ Fill in your measured values from `model_train.ipynb` outputs. CICIDS 2017 IDS benchmarks typically achieve 97–99% accuracy; UNSW-NB15 is harder (~90–95%).

---

## 🌐 Web Application

The Flask app provides a clean interface to:
1. **Upload** a CSV of network flow features
2. **Visualize** the generated RGB and Grayscale images
3. **Get a prediction** — BENIGN or ATTACK — from the ensemble

Endpoints:
- `GET /` — Home page
- `POST /process` — Upload CSV → returns prediction + image
- `GET /download` — Download generated grayscale image

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | TensorFlow / Keras |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Image Processing | Matplotlib, SciPy (interp2d, RegularGridInterpolator) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Web Backend | Flask |
| Model Persistence | Joblib, Keras `.h5` |
| Datasets | CICIDS 2017/2018, UNSW-NB15 |

---

## 🗺️ Future Work

- [ ] Real-time packet capture and inference (PyShark / Scapy integration)
- [ ] Explainability layer — Grad-CAM visualization on predicted images
- [ ] Transformer-based encoder (ViT) to replace CNN backbone
- [ ] Docker containerization for one-command deployment
- [ ] REST API with authentication for enterprise integration
- [ ] Multi-class attack classification (DDoS, Probe, R2L, U2R)

---

## 📚 Dataset References

- **CICIDS 2017/2018** — Canadian Institute for Cybersecurity, University of New Brunswick. Covers DoS, DDoS, Brute Force, Web attacks.
- **UNSW-NB15** — UNSW Canberra. Modern hybrid dataset with real attack traffic and synthetic normal traffic.

---

## 👤 Author

**Mohammed Fahad Altamash**  
Bangalore, India  
[GitHub](https://github.com/fahad03-mfa)

---

> If this project helped you, consider giving it a ⭐ — it helps others find it!
