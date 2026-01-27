import cv2

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def start_cam(camera_index=0):
    """Ouvre la caméra webcam."""
    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {camera_index}")
    return video


def detect_faces(frame):
    """Détecte tous les visages dans une image et retourne les bboxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return FACE_CASCADE.detectMultiScale(gray, 1.1, 4)


def extract_face(frame):
    """Extrait et normalise le visage le plus grand en image 48×48 en niveaux de gris."""
    faces = detect_faces(frame)
    if len(faces) == 0:
        return None
    
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = frame[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray_face, (48, 48))


if __name__ == "__main__":
    cam = start_cam()
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            face = extract_face(frame)
            if face is not None:
                face_large = cv2.resize(face, (192, 192), interpolation=cv2.INTER_NEAREST)
                face_bgr = cv2.cvtColor(face_large, cv2.COLOR_GRAY2BGR)
                
                h_frame, w_frame = frame.shape[:2]
                face_h, face_w = face_bgr.shape[:2]
                
                if face_h <= h_frame and face_w <= w_frame:
                    y_offset, x_offset = 10, w_frame - face_w - 10
                    frame[y_offset:y_offset+face_h, x_offset:x_offset+face_w] = face_bgr
                    cv2.putText(frame, "Face 48x48", (x_offset, face_h+30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Webcam + Face", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()