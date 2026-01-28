import cv2
import numpy as np
import torch
import argparse
from src.utils.model_loader import load_model, get_class_names
from src.utils.prediction import predict_emotion

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def start_cam(camera_index=0):
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {camera_index}")
    return video

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FACE_CASCADE.detectMultiScale(gray, 1.1, 4)

def extract_face(frame):
    faces = detect_faces(frame)
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = frame[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_48 = cv2.resize(gray_face, (48, 48))
    return face_48, (x, y, w, h)

if __name__ == "__main__":
    # Parser pour les arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Détection d'émotions en temps réel")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"],
                       help="Type de modèle à utiliser: 'cnn' ou 'resnet' (défaut: cnn)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Modèle: {args.model.upper()}")
    
    # Choisir le fichier de modèle selon le type
    if args.model == "resnet":
        model_path = "emotion_resnet_best.pt"
    else:
        model_path = "emotion_best.pt"
    
    model = load_model(model_path, model_type=args.model, num_classes=7, device=device)
    id_to_class = get_class_names()
    
    history = []
    cam = start_cam()
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            face_48, bbox = extract_face(frame)
            
            if face_48 is not None:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                pred_id, conf, probs = predict_emotion(model, face_48, device)
                
                history.append(probs)
                if len(history) > 10:
                    history.pop(0)
                
                avg_probs = np.mean(history, axis=0)
                smoothed_pred_id = int(np.argmax(avg_probs))
                smoothed_conf = float(avg_probs[smoothed_pred_id])
                emotion_label = id_to_class[smoothed_pred_id]
                
                face_large = cv2.resize(face_48, (192, 192), interpolation=cv2.INTER_NEAREST)
                face_bgr = cv2.cvtColor(face_large, cv2.COLOR_GRAY2BGR)
                
                h_frame, w_frame = frame.shape[:2]
                face_h, face_w = face_bgr.shape[:2]
                
                if face_h <= h_frame and face_w <= w_frame:
                    y_offset, x_offset = 10, w_frame - face_w - 10
                    frame[y_offset:y_offset+face_h, x_offset:x_offset+face_w] = face_bgr
                    cv2.putText(frame, "Face 48x48", (x_offset, face_h+30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                label_text = f"{emotion_label} ({smoothed_conf*100:.1f}%)"
                cv2.putText(frame, label_text, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if len(history) > 0:
                    history = []
            
            cv2.imshow("Emotion Detection - Real-time", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
