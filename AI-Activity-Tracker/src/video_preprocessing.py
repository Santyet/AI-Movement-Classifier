import os
import cv2

RAW_VID_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'videos')
FRAME_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'frames')

def extract_frames(label, fps=5):
    """Extrae frames de todos los videos de una actividad (label)."""
    label_in = os.path.join(RAW_VID_DIR, label)
    label_out = os.path.join(FRAME_DIR, label)
    os.makedirs(label_out, exist_ok=True)
    for vid in os.listdir(label_in):
        cap = cv2.VideoCapture(os.path.join(label_in, vid))
        count, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % fps == 0:
                fname = f"{os.path.splitext(vid)[0]}_frame{saved:04d}.jpg"
                cv2.imwrite(os.path.join(label_out, fname), frame)
                saved += 1
            count += 1
        cap.release()
        print(f"[{label}] {vid} → {saved} frames")

if __name__ == "__main__":
    for activity in os.listdir(RAW_VID_DIR):
        extract_frames(activity, fps=10)
    print("Extracción de frames completada.")
