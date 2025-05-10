import os
import csv
import cv2
import mediapipe as mp

FRAME_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'frames')
FEATURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features')
os.makedirs(FEATURE_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
landmark_names = [f"{lm.name}_{coord}" for lm in mp_pose.PoseLandmark for coord in ('x','y','z','v')]

def extract_features():
    """Recorre cada frame, extrae landmarks y guarda en un CSV único."""
    outfile = os.path.join(FEATURE_DIR, 'pose_features.csv')
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        # Cabecera: label + todos los coords
        writer.writerow(['activity'] + landmark_names)
        for activity in os.listdir(FRAME_DIR):
            folder = os.path.join(FRAME_DIR, activity)
            for imgf in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, imgf))
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    continue
                row = [activity]
                for lm in results.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                writer.writerow(row)
    print("Extracción de features completada.")

if __name__ == "__main__":
    extract_features()
