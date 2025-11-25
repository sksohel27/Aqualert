# Aqualert: AI-Powered Urban Water Hazard Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![React](https://img.shields.io/badge/React-18.x-green.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-blueviolet.svg)](https://flask.palletsprojects.com/)

> **Updated for 2025**: Enhanced with improved model versioning, dataset expansion, and deployment optimizations for real-time urban monitoring.

## Overview

**Aqualert** is a cutting-edge, full-stack AI application for detecting and alerting on urban water hazards, including **urban flooding**, **drainage blockages**, and **pond pollution**. Developed for the **Smart India Hackathon (SIH) 2025**, it combines **Convolutional Neural Networks (CNNs)** with a modern web interface to enable users to upload images and receive instant classifications and safety recommendations.

Powered by **TensorFlow/Keras** for the ML backend and **React/Flask** for the frontend, Aqualert processes images in real-time, achieving **83-89% accuracy** on a curated dataset of 2,577+ environmental images. Ideal for civic authorities, environmental NGOs, and citizens to promote proactive urban sustainability.

### Core Technologies
- **ML**: Custom CNN for multi-class image classification (3 classes).
- **Backend**: Flask API for model inference and database integration (SQLite/MySQL via Node.js helpers).
- **Frontend**: React for responsive UI, image upload, and visualization of predictions.
- **Data Pipeline**: TensorFlow `tf.data` for efficient loading, augmentation, and batching.

## Key Features
- **Real-Time Hazard Detection**: Upload photos → AI classifies → Generate alerts (e.g., "High-risk flooding detected – Evacuate low-lying areas").
- **Data Augmentation**: Random flips/rotations to handle imbalanced datasets and improve robustness.
- **Model Evaluation & Visualization**: Training history plots, confidence scores, and sample predictions.
- **Auto-Versioned Models**: Saves trained models incrementally (e.g., `Model_3/1/`) for easy rollback/experimentation.
- **Hybrid Stack**: Python for ML training, Node.js for server/DB, React for UI.
- **Deployment-Ready**: Optimized for local dev or cloud (e.g., Heroku/AWS) with `requirements.txt` and `package.json`.

## Project Structure

```
Aqualert/
├── backend/                     # Primary Python/ML backend
│   ├── __pycache__/             # Compiled Python
│   ├── main.py                  # Flask entrypoint (API routes for predictions)
│   ├── package-lock.json        # NPM lock (for hybrid Node deps)
│   ├── package.json             # NPM deps (e.g., for server.js)
│   ├── requirements.txt         # Python deps (tensorflow, flask, etc.)
│   ├── server.js                # Node.js server (optional hybrid API)
│   └── README.md                # Backend docs
│
├── frontend/                    # React frontend (alternative/modular setup)
│   ├── __pycache__/             # Compiled files
│   ├── main.python-311.pyc      # Cached Python (if integrated)
│   ├── public/
│   │   ├── cblogo.png           # App logo
│   │   ├── index.html           # Static entry
│   │   └── robots.txt           # Web crawler config
│   ├── src/
│   │   ├── App.js               # React root
│   │   ├── App.test.js          # Tests
│   │   ├── bg.png               # Background asset
│   │   ├── cblogo.png           # Logo duplicate
│   │   ├── home.js              # Home component (upload/predict UI)
│   │   ├── index.css            # Styles
│   │   ├── index.js             # React bootstrap
│   │   ├── reportWebVitals.js   # Perf metrics
│   │   ├── README.md            # Frontend docs
│   │   ├── package-lock.json
│   │   └── package.json
│   └── server/                  # Nested server (DB + models)
│       ├── __pycache__/
│       ├── database/
│       │   └── conn.js          # DB connection (e.g., Sequelize)
│       └── model/
│           ├── post_model.js    # DB schema for alerts/posts
│           └── post_model.js    # (Duplicate? – likely for posts/alerts)
│
├── data/                        # Dataset (gitignore large files)
│   ├── Urban_flooding/          # ~927 images
│   ├── drainage/                # ~616 images
│   └── ponds/                   # ~459 images
│
├── Model/
│   └── Model_3/                 # CNN model saves
│       ├── 1/                   # Version 1 (auto-incremented)
│       │   └── assets/          # TF SavedModel assets
│       └── classifier_model2.h5 # HDF5 model file (legacy format)
│
├── model.py                     # Core ML script (training/inference)
└── .gitignore                   # Exclude data/models/__pycache__
```

## Quick Start

### Prerequisites
- Python 3.8+ (for ML)
- Node.js 18+ (for frontend/server)
- Git

### Installation
1. **Clone & Setup**:
   ```
   git clone https://github.com/yourusername/aqualert.git
   cd aqualert
   ```

2. **Backend (ML + Flask)**:
   ```
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   # Download dataset to ./data/ (subfolders by class)
   ```

3. **Frontend (React)**:
   ```
   cd frontend/src  # Or backend/src if integrated
   npm install
   ```

4. **Database** (Optional):
   - Install MySQL/SQLite.
   - Run `node server/database/conn.js` to init.

### Training the Model
1. Edit `data_dir` in `model.py` to point to `./data/`.
2. Run:
   ```
   python model.py
   ```
   - Partitions data (80% train, 10% val, 10% test).
   - Trains CNN for 100 epochs (~2-3 hours on CPU; faster on GPU).
   - Generates plots (accuracy/loss) and saves model to `./Model/Model_3/`.

   **Sample Metrics**:
   | Split   | Accuracy | Loss  |
   |---------|----------|-------|
   | Train   | 89.4%   | 0.29 |
   | Val     | 84.0%   | 0.39 |
   | Test    | 83.0%   | 0.36 |

### Running the App
1. **Start Backend**:
   ```
   python main.py  # Flask on http://localhost:5000
   ```
2. **Start Frontend**:
   ```
   npm start  # React on http://localhost:3000
   ```
3. **Test**: Upload an image via UI → View classification (e.g., "Drainage Issue – 92% confidence") + alert.

### Inference Example
```python
import tensorflow as tf
from model import predict, class_names  # Assuming utils in model.py

model = tf.keras.models.load_model('Model/Model_3/classifier_model2.h5')
img_path = 'path/to/test_image.jpg'
predicted_class, confidence = predict(model, img_path)
print(f"Hazard: {predicted_class} ({confidence:.2f}%)")
```

## Model Architecture
- **Input**: 256x256x3 RGB images.
- **Layers**: 6x Conv2D (32-64 filters, 3x3, ReLU) + MaxPooling2D → Flatten → Dense(64, ReLU) → Dense(3, Softmax).
- **Optimizer**: Adam | **Loss**: SparseCategoricalCrossentropy.
- **Total Params**: ~184K (lightweight for edge deployment).
- **Augmentation**: Horizontal/vertical flips + 20° rotations.

For improvements: Add transfer learning (e.g., ResNet) or class weights for imbalance.

## Contributing
- Fork → Branch (`git checkout -b feature/your-feature`) → Commit → PR.
- Issues: Report bugs or suggest dataset expansions (e.g., add "sewage overflow").

## License
MIT License – Free to use, modify, and distribute. See [LICENSE](LICENSE).

## Acknowledgments
- **SIH 2025** for the urban sustainability challenge.
- TensorFlow community for CNN tools.
- Open-source datasets for environmental imagery.

---

⭐ Star if this helps urban monitoring! Questions? [Open an issue](https://github.com/yourusername/aqualert/issues). Let's build smarter cities! 🌊
