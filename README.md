# IoT & 5G Intrusion Detection System using Deep Learning

## Overview
A deep-learning based intrusion detection system designed for IoT and 5G environments. The project transforms network-flow features into RGB and grayscale image representations and uses convolutional neural networks to classify traffic as BENIGN or ATTACK.

## Key Features
- Network intrusion detection for IoT and 5G traffic
- CICIDS and UNSW-NB15 dataset support
- Data preprocessing and feature scaling
- Class imbalance handling using SMOTE and undersampling
- Feature-to-image conversion (RGB and Grayscale)
- CNN-based classification models
- Flask web application for inference
- Ensemble prediction strategy

## Project Architecture
```text
Network Traffic CSV
        |
        v
Data Preprocessing
(Scaling + Cleaning)
        |
        v
Feature Transformation
(RGB / Grayscale Images)
        |
        v
Deep Learning Models
        |
        v
Ensemble Prediction
        |
        v
Flask Web Application
```

## Datasets
- CICIDS 2017
- CICIDS 2018
- UNSW-NB15

These datasets contain normal and malicious network traffic samples used for cybersecurity research and intrusion detection.

## Methodology
### 1. Data Preprocessing
- Feature scaling using MinMaxScaler
- Label encoding
- Binary classification conversion (Attack vs Benign)

### 2. Class Balancing
- SMOTE oversampling
- Random undersampling

### 3. Feature Visualization
Network-flow features are transformed into:
- RGB images
- Grayscale images

### 4. Deep Learning Classification
Two CNN models are used:
- RGB Model
- Grayscale Model

### 5. Ensemble Learning
Predictions from both models are combined to improve robustness.

## Repository Structure
```text
Intrusions/
├── app/
│   ├── preprocessing.py
│   ├── image_converter.py
│   └── main.py
├── templates/
├── static/
├── models/
├── notebooks/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Project
```bash
git clone https://github.com/fahad03-mfa/Intrusions.git
cd Intrusions
pip install -r requirements.txt
python main.py
```

## Results
Add your measured values here.

| Metric | RGB Model | Grayscale Model | Ensemble |
|----------|----------|----------|----------|
| Accuracy | XX.XX% | XX.XX% | XX.XX% |
| Precision | XX.XX% | XX.XX% | XX.XX% |
| Recall | XX.XX% | XX.XX% | XX.XX% |
| F1 Score | XX.XX% | XX.XX% | XX.XX% |

## Screenshots
Add screenshots of:
- Home page
- CSV upload page
- Generated RGB image
- Generated Grayscale image
- Prediction page

## Future Improvements
- Real-time network monitoring
- Explainable AI integration
- Transformer-based intrusion detection
- Cloud deployment
- Docker support
- REST API endpoints

## Tech Stack
- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Author
Mohammed Fahad Altamash

If you found this project useful, consider giving the repository a star.
