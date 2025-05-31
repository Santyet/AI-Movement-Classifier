import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import sys
from sklearn.preprocessing import StandardScaler

# Cargar modelo
try:
    with open('models/modelo_movimientos.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo modelo_movimientos.pkl")
    sys.exit(1)

# Cargar etiquetas (lista simple como ['caminar', 'saltar', ...])
try:
    with open('models/label_encoder.pkl', 'rb') as f:
        etiquetas = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo label_encoder.pkl")
    sys.exit(1)

# Cargar el scaler usado durante el entrenamiento
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler cargado correctamente")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo scaler.pkl")
    print("📝 Es necesario guardar el scaler durante el entrenamiento")
    scaler = None

print("✅ Modelo y etiquetas cargados:", etiquetas)

# Inicializar MediaPipe Pose
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

# Captura de cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara")
    sys.exit(1)

# Configurar propiedades de la cámara para mejor rendimiento
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir buffer para menor latencia

# Cola para almacenar 5 frames
frames_queue = deque(maxlen=5)

# Cola para almacenar datos para entrenar el scaler (solo si no se cargó uno)
data_for_scaler = deque(maxlen=50)  # Almacenar 50 secuencias para entrenar scaler
if scaler is None:
    scaler = StandardScaler()
    scaler_trained = False
else:
    scaler_trained = True


# Control de saltar frames
frame_skip = 2
frame_count = 0

# Variable para controlar la ventana
window_name = 'Prediccion de movimiento'
window_created = False

def extraer_landmarks(results):
    """Extrae los landmarks de pose de MediaPipe"""
    if results.pose_landmarks:
        return np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()
    return None

def crear_ventana():
    """Crea la ventana de OpenCV una sola vez"""
    global window_created
    if not window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_created = True

def cleanup():
    """Función para limpiar recursos"""
    print("🔁 Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

try:
    print("🎥 Iniciando captura de video. Presiona 'q' para salir.")
    
    # Crear ventana una sola vez
    crear_ventana()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara")
            break

        # Saltar algunos frames para suavizar
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame_count += 1

        # Rotar si es horizontal (comentado para mejor rendimiento)
        # h, w, _ = frame.shape
        # if w > h:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Procesar con MediaPipe (optimizado)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Mejora rendimiento
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Dibujar landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )

        # Extraer y guardar landmarks
        landmarks = extraer_landmarks(results)
        if landmarks is not None:
            frames_queue.append(landmarks)

            # Hacer predicción cuando tengamos 5 frames
            if len(frames_queue) == 5:
                try:
                    # Concatenar la secuencia de landmarks
                    secuencia = np.concatenate(frames_queue).reshape(1, -1)
                    
                    # IMPORTANTE: Aplicar la misma normalización que durante el entrenamiento
                    # Escalar cada frame individualmente y luego concatenarlos
                    secuencia_escalada = []

                    for frame_landmarks in frames_queue:
                        # Cada frame tiene shape (99,)
                        frame_landmarks_scaled = scaler.transform(frame_landmarks.reshape(1, -1)).flatten()
                        secuencia_escalada.append(frame_landmarks_scaled)

                    # Concatenar los frames escalados en la secuencia final de 495 features
                    secuencia_normalizada = np.concatenate(secuencia_escalada).reshape(1, -1)

                    
                    # Hacer predicción con datos normalizados
                    pred = model.predict(secuencia_normalizada)
                    label = etiquetas[pred[0]]
                    
                    # Mostrar predicción en el frame con mejor formato
                    cv2.rectangle(frame, (5, 5), (400, 100), (0, 0, 0), -1)  # Fondo negro
                    cv2.putText(frame, f'Movimiento: {label}', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Mostrar confianza si el modelo lo soporta
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(secuencia_normalizada)
                        confidence = np.max(proba) * 100
                        cv2.putText(frame, f'Confianza: {confidence:.1f}%', (10, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                except Exception as e:
                    print(f"⚠️ Error al predecir: {str(e)}")
                    cv2.rectangle(frame, (5, 5), (350, 50), (0, 0, 0), -1)  # Fondo negro
                    cv2.putText(frame, 'Error en prediccion', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Agregar instrucciones en pantalla
        cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar frame en la ventana existente
        cv2.imshow(window_name, frame)

        # Verificar teclas presionadas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' o ESC
            print("🛑 Salida solicitada por el usuario.")
            break
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            # La ventana fue cerrada
            print("🛑 Ventana cerrada por el usuario.")
            break

except KeyboardInterrupt:
    print("🛑 Interrupción manual detectada (Ctrl+C).")
except Exception as e:
    print(f"❌ Error inesperado: {str(e)}")
finally:
    cleanup()
    print("✅ Programa terminado correctamente.")