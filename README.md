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

### FaceForensics++ (Primary Dataset)
1. Fill out the access request form at: https://github.com/ondyari/FaceForensics
2. Once approved, download the following c23 compression folders:
   - `original_sequences/youtube/c23/videos/`
   - `manipulated_sequences/Deepfakes/c23/videos/`
   - `manipulated_sequences/Face2Face/c23/videos/`
   - `manipulated_sequences/FaceSwap/c23/videos/`
   - `manipulated_sequences/NeuralTextures/c23/videos/`
3. Organize everything under one root folder, e.g. `FF_Dataset/`

### Celeb-DF v2 (Cross-Dataset Validation)
Used for cross-dataset generalization testing in the final project.
- Download from Kaggle: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
- Place real videos in `Celeb_DF/real/` and fake videos in `Celeb_DF/fake/`

---

## Additional File Required

Download the dlib 68-point facial landmark predictor:
- Link: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Extract and place `shape_predictor_68_face_landmarks.dat` inside your `FF_Dataset/` folder

---

## Pretrained Models

Download all 4 pretrained models from Google Drive and place them inside your `FF_Dataset/` folder:

| Model File | Description | Download |
|------------|-------------|----------|
| `xception_model.h5` | Xception (primary deep learning model) | [Download](https://drive.google.com/file/d/1GOPA14OhYCVm40J0VV2oa1Jl3O5_FfKw/view?usp=sharing) |
| `efficientnet_model.h5` | EfficientNet-B0 with Spatial Attention | [Download](https://drive.google.com/file/d/1MVWjkzzfjYp2ek8dibgDfUXdJXW2r3Gd/view?usp=sharing) |
| `svm_model.pkl` | FFT + SVM baseline | [Download](https://drive.google.com/file/d/1GHKqGyTXiUJyFXV1LOQK0nUyvomYTou_/view?usp=sharing) |
| `rf_model.pkl` | dlib + Random Forest baseline | [Download](https://drive.google.com/file/d/1H-1ciqhQRTtgLheivX31K25MlRWTz3mN/view?usp=sharing) |

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
- Equal Weighted Ensemble (0.5/0.5) — optimal weights found via sensitivity analysis
- Ensemble Weight Sensitivity Analysis
- Background Masking for aggressive face cropping
- Grad-CAM explainability (correct predictions + failed cases)
- Per-manipulation type analysis
- Cross-dataset validation on Celeb-DF v2
- Grad-CAM on Celeb-DF failed cases
- Real-time Grad-CAM video overlay

> **Smart Skip Logic:** Training cells automatically skip if a saved model already exists. To retrain, delete the corresponding file.

### Option 2 - Demo Script
After models are in place, run the demo on any video:
```bash
python demo.py
```
The demo uses the Equal Weighted Ensemble (0.5/0.5) to classify a video as **REAL** or **FAKE** with a confidence score.

---

## Results Summary

| Model | Accuracy | AUC |
|-------|----------|-----|
| FFT + SVM | 75.33% | 0.8124 |
| dlib + Random Forest | 76.74% | 0.8694 |
| Xception | 92.67% | 0.9881 |
| EfficientNet-B0 + Spatial Attention | 89.33% | 0.9611 |
| Equal Ensemble (0.5/0.5) | **95.33%** | **0.9767** |
| Celeb-DF Cross-Dataset | 75.00% | — |

---

## Repository Contents

| File | Description |
|------|-------------|
| `deepfake_detection.ipynb` | Main notebook - all 5 models, analysis, and Grad-CAM |
| `demo.py` | Real-time video classification demo |
| `efficientnet_confusion_matrix.png` | EfficientNet-B0 confusion matrix |
| `ensemble_confusion_matrix.png` | Equal Ensemble (0.5/0.5) confusion matrix |
| `xception_confusion_matrix.png` | Xception confusion matrix |
| `svm_confusion_matrix.png` | FFT+SVM confusion matrix |
| `rf_confusion_matrix.png` | Random Forest confusion matrix |
| `gradcam_heatmaps.png` | Grad-CAM on correct predictions |
| `gradcam_failed_cases.png` | Grad-CAM on FF++ failed predictions |
| `gradcam_celebdf_failed.png` | Grad-CAM on Celeb-DF failed predictions |
| `model_comparison.png` | Accuracy comparison across all models |
| `face_extraction_sample.png` | Sample MTCNN face extraction output |
| `xception_training_curves.png` | Xception training accuracy/loss curves |
| `ensemble_sensitivity.png` | Ensemble weight sensitivity analysis graph |
| `background_masking_sample.png` | Background masking comparison |
| `gradcam_real_output.mp4` | Real-time Grad-CAM overlay on real video |
| `gradcam_fake_output.mp4` | Real-time Grad-CAM overlay on fake video |

> **Note:** All pretrained models are hosted on Google Drive. Dataset videos and extracted face images are not included due to size.