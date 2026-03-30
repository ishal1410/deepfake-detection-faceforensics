# Deepfake Detection - FaceForensics++

**Course:** AI688-001 Image and Vision Computing, Spring 2026  
**Professor:** Reda Nacif Elalaoui  
**Institution:** Long Island University, Brooklyn  
**Group 2:** Vishal Patel | Saunil Patel

---

## Requirements

- Python 3.10+
- NVIDIA GPU recommended (tested on RTX 4060)
- ~10GB free disk space for dataset

Install dependencies:
```bash
pip install tensorflow opencv-python dlib mtcnn scikit-learn matplotlib seaborn numpy pandas
```

---

## Dataset

This project uses the **FaceForensics++ Low Quality (c23)** dataset.

1. Fill out the access request form at: https://github.com/ondyari/FaceForensics
2. Once approved, you will receive a download script via email
3. Use the download script to download the following c23 compression folders:
   - `original_sequences/youtube/c23/videos/`
   - `manipulated_sequences/Deepfakes/c23/videos/`
   - `manipulated_sequences/Face2Face/c23/videos/`
   - `manipulated_sequences/FaceSwap/c23/videos/`
   - `manipulated_sequences/NeuralTextures/c23/videos/`
4. Organize everything under one root folder, e.g. `FF_Dataset/`

---

## Additional File Required

Download the dlib 68-point facial landmark predictor:
- Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Extract the `.bz2` file and place `shape_predictor_68_face_landmarks.dat` inside your `FF_Dataset/` folder

---

## Pretrained Models

Download all 4 pretrained models from Google Drive and place them inside your `FF_Dataset/` folder:

| Model File | Description | Download |
|------------|-------------|----------|
| `xception_model.h5` | Xception (primary deep learning model) | [Download](https://drive.google.com/file/d/1GOPA14OhYCVm40J0VV2oa1Jl3O5_FfKw/view?usp=sharing) |
| `efficientnet_model.h5` | EfficientNet-B0 with Spatial Attention | [Download](https://drive.google.com/file/d/1MVWjkzzfjYp2ek8dibgDfUXdJXW2r3Gd/view?usp=sharing) |
| `svm_model.pkl` | FFT + SVM baseline | [Download](https://drive.google.com/file/d/1GHKqGyTXiUJyFXV1LOQK0nUyvomYTou_/view?usp=sharing) |
| `rf_model.pkl` | dlib + Random Forest baseline | [Download](https://drive.google.com/file/d/1H-1ciqhQRTtgLheivX31K25MlRWTz3mN/view?usp=sharing) |

> If you prefer to train the models yourself, skip this step and run the full notebook - models will be trained and saved automatically.

---

## Setup

**1. Clone this repository:**
```bash
git clone https://github.com/ishal1410/deepfake-detection-faceforensics.git
cd deepfake-detection-faceforensics
```

**2. Download pretrained models:**  
Download all 4 models from the links above and place them in your `FF_Dataset/` folder.

**3. Open the notebook:**  
Open `deepfake_detection.ipynb` in VS Code or Jupyter Notebook

**4. Update the dataset path:**  
In **Cell 2**, change the path to match your local setup:
```python
DATASET_PATH = r"C:\your\path\to\FF_Dataset"
```

---

## How to Run

### Option 1 - Full Pipeline (Notebook)
Run all cells in `deepfake_detection.ipynb` from top to bottom. The notebook includes:
- Face extraction and alignment using MTCNN
- FFT + SVM baseline model
- dlib + Random Forest baseline model
- Xception deep learning model (two-stage fine-tuning)
- EfficientNet-B0 with Spatial Attention Module
- Trust-Aware Weighted Ensemble
- Grad-CAM explainability (correct predictions + failed cases)
- Per-manipulation type analysis

> **Smart Skip Logic:** Training cells automatically skip if a saved model already exists in your `FF_Dataset/` folder. To retrain any model from scratch, delete the corresponding file:
> - `svm_model.pkl` - FFT + SVM
> - `rf_model.pkl` - Random Forest
> - `xception_model.h5` - Xception
> - `efficientnet_model.h5` - EfficientNet-B0

### Option 2 - Demo Script
After models are in place, run the demo on any video:
```bash
python demo.py
```
The demo uses the Trust-Aware Weighted Ensemble to classify a video as **REAL** or **FAKE** with a confidence score.

---

## Repository Contents

| File | Description |
|------|-------------|
| `deepfake_detection.ipynb` | Main notebook - all 5 models, analysis, and Grad-CAM |
| `demo.py` | Real-time video classification demo |
| `efficientnet_confusion_matrix.png` | EfficientNet-B0 confusion matrix |
| `ensemble_confusion_matrix.png` | Ensemble confusion matrix |
| `xception_confusion_matrix.png` | Xception confusion matrix |
| `svm_confusion_matrix.png` | FFT+SVM confusion matrix |
| `rf_confusion_matrix.png` | Random Forest confusion matrix |
| `gradcam_heatmaps.png` | Grad-CAM on correct predictions |
| `gradcam_failed_cases.png` | Grad-CAM on failed predictions |
| `model_comparison.png` | Accuracy comparison across all models |
| `face_extraction_sample.png` | Sample MTCNN face extraction output |
| `xception_training_curves.png` | Xception training accuracy/loss curves |

> **Note:** All pretrained models are hosted on Google Drive. Dataset videos and extracted face images are not included due to size.