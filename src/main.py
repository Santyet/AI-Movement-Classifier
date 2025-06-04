import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import sys




def features_de_ventana(frames_list):
    """
    Recibe lista de `window_size` arrays de forma (99,), en pixeles
    Devuelve vector de 396 features (media/var y media/std de vel).
    """
    window_size = len(frames_list)
    n_landmarks = 33
    arr = np.stack(frames_list, axis=0).reshape(window_size, n_landmarks, 3)

    feats = []
    
    for j in range(n_landmarks):
        for a in range(3):
            vals = arr[:, j, a]
            feats.append(np.mean(vals))
            feats.append(np.var(vals))
    
    diffs = np.diff(arr, axis=0)
    for j in range(n_landmarks):
        for a in range(3):
            vel = diffs[:, j, a]
            feats.append(np.mean(vel))
            feats.append(np.std(vel))

    return np.array(feats)  

def extraer_landmarks(results):
    """
    Extrae 33 landmarks × (x,y,z) en forma normalized (0–1).
    Retorna array (99,) o None si no hay landmarks.
    """
    if results.pose_landmarks:
        return np.array([[lmk.x, lmk.y, lmk.z]
                         for lmk in results.pose_landmarks.landmark]).flatten()
    return None

def bbox_height(frame_lmks):
    """
    Calcula la altura en píxeles del cuerpo: (y_max - y_min) 
    a partir de un array de 99 dims donde los Y están en índices 1, 4, 7, …
    """
    ys = frame_lmks[1::3]  
    return max(ys) - min(ys)

def estimar_delta_bbox(frames_list):
    """
    % de cambio de altura entre el primer y el último frame,
    usando bbox_height. Retorna (hN - h0) / h0.
    """
    h0 = bbox_height(frames_list[0])
    hN = bbox_height(frames_list[-1])
    if h0 == 0:
        return 0.0
    return (hN - h0) / h0  

def crear_ventana_opencv():
    global window_created
    if not window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_created = True

def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    pose.close()




try:
    model = joblib.load('models/modelo_movimientos.pkl')
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo modelo_movimientos.pkl")
    sys.exit(1)

try:
    le = joblib.load('models/label_encoder.pkl')
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo label_encoder.pkl")
    sys.exit(1)

try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler cargado correctamente")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo scaler.pkl")
    print("📝 Es necesario guardar el scaler durante el entrenamiento")
    sys.exit(1)

print("✅ Modelo y LabelEncoder cargados")

pca_trained = model.named_steps['pca']
clf_trained = model.named_steps['clf']

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

window_size = 10
frames_queue = deque(maxlen=window_size)

frame_skip = 2
frame_count = 0

window_name = 'Prediccion de movimiento'
window_created = False

try:
    print("🎥 Iniciando captura de video. Presiona 'q' para salir.")
    crear_ventana_opencv()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara")
            break

        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )

        landmarks_norm = extraer_landmarks(results)
        if landmarks_norm is not None:
            
            landmarks_px = landmarks_norm.copy()
            for i in range(33):
                landmarks_px[i*3 + 0] = landmarks_norm[i*3 + 0] * w   
                landmarks_px[i*3 + 1] = landmarks_norm[i*3 + 1] * h   
                
            

            frames_queue.append(landmarks_px)

            
            if len(frames_queue) == window_size and frame_count % 5 == 0:
                try:
                    
                    feats_ventana = features_de_ventana(list(frames_queue))  

                    X_scaled = scaler.transform(feats_ventana.reshape(1, -1))  
                    X_pca = pca_trained.transform(X_scaled)  

                    proba = clf_trained.predict_proba(X_pca)[0]                  
                    idx_pred = np.argmax(proba)
                    label = le.inverse_transform([idx_pred])[0]
                    delta_h = estimar_delta_bbox(list(frames_queue))
                    umbral_h = 0.08  
                    override = None
                    if delta_h > umbral_h:
                        override = 'caminar-adelante'
                    
                    elif delta_h < -umbral_h:
                        override = 'caminar-atrás'

                    if override is not None and proba[idx_pred] < 0.70:
                        label = override

                    cv2.rectangle(frame, (5, 5), (500, 140), (0, 0, 0), -1)
                    cv2.putText(frame, f'Movimiento: {label}', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    for i, cls in enumerate(le.classes_):
                        cv2.putText(frame, f'{cls}: {proba[i]*100:4.1f}%', (10, 65 + 20*i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                except Exception as e:
                    print(f"⚠️ Error al predecir: {e}")
                    cv2.rectangle(frame, (5, 5), (350, 50), (0, 0, 0), -1)
                    cv2.putText(frame, 'Error en prediccion', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
        cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("🛑 Salida solicitada por el usuario.")
            break
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Ventana cerrada por el usuario.")
            break

except KeyboardInterrupt:
    print("🛑 Interrupción manual detectada (Ctrl+C).")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
finally:
    cleanup()
    print("✅ Programa terminado correctamente.")
