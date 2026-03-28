import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mtcnn import MTCNN

# Paths
DATASET_PATH = r"C:\Users\vp141\Downloads\FF_Dataset"
xception_model_path = os.path.join(DATASET_PATH, "xception_model.h5")
efficientnet_model_path = os.path.join(DATASET_PATH, "efficientnet_model.h5")

# Load models
print("Loading models...")
xception_model = tf.keras.models.load_model(xception_model_path)
efficientnet_model = tf.keras.models.load_model(efficientnet_model_path)
detector = MTCNN()
print("Models loaded!")

def predict_video(video_path):
    print(f"\nAnalyzing: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, 10, dtype=int)
    
    predictions = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        
        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (224, 224))
            face_normalized = face / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            
            prob_xception = xception_model.predict(face_input, verbose=0)[0][0]
            prob_eff = efficientnet_model.predict(face_input, verbose=0)[0][0]
            prob_ensemble = (0.6 * prob_xception) + (0.4 * prob_eff)
            predictions.append(prob_ensemble)
    
    cap.release()
    
    if not predictions:
        print("No faces detected!")
        return
    
    avg_prob = np.mean(predictions)
    label = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = avg_prob * 100 if label == "FAKE" else (1 - avg_prob) * 100
    
    print(f"Faces analyzed: {len(predictions)}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Fake probability: {avg_prob*100:.2f}%")

# Run
video_path = input("Enter video path: ")
predict_video(video_path)