import cv2
import os
import mediapipe as mp
import pandas as pd

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Ruta base de los videos organizados por clase
video_base_path = 'data/raw/videos'
# Nueva ruta para guardar los frames procesados
images_base_path = 'data/processed/images'

# Ruta para guardar los datos en CSV y Excel
csv_save_path = './data/processed/datos.csv'
excel_save_path = './data/processed/datos.xlsx'

# Lista para almacenar los datos
all_data = []

# Recorrer todas las carpetas y archivos de video
for root, dirs, files in os.walk(video_base_path):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(root, file)
            label = os.path.basename(root)  # El nombre de la carpeta es la etiqueta
            
            # Leer el video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'Procesando: {video_path} - FPS: {fps}')

            frame_skip = 2  # Procesar 1 de cada 2 frames (ajusta según sea necesario)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar solo cada 'frame_skip' frames
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                frame_count += 1

                # Verificar si la imagen está horizontal y rotarla si es necesario
                height, width, _ = frame.shape
                if width > height:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    frame_data = {
                        'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                        'label': label
                    }

                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        frame_data[f'landmark_{i}_x'] = landmark.x
                        frame_data[f'landmark_{i}_y'] = landmark.y
                        frame_data[f'landmark_{i}_z'] = landmark.z

                    all_data.append(frame_data)

            cap.release()

# Crear DataFrame y guardar
df = pd.DataFrame(all_data)
os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
df.to_csv(csv_save_path, index=False)
df.to_excel(excel_save_path, index=False)

print(f"Datos guardados en {csv_save_path} y {excel_save_path}")