# Real-Time Human Activity Classification System

## 👥 Team Members

* Juan José Díaz
* Santiago Espinosa
* Ana Londoño

**Final Project – Artificial Intelligence 1**

This repository contains a complete pipeline to recognize five human actions in real time using MediaPipe Pose and an XGBoost classifier.

---

**Videos**

[Demo Video](https://www.youtube.com/watch?v=Pecin6A7uNE)
[Explanatory Video](https://youtu.be/B9QzLmgQlKk)

---

## 📂 Project Structure


## Brief Descriptions

- **data/raw/videos/**  
  Contains subfolders for each action (`caminar-adelante`, `caminar-atras`, `girar`, `pararse`, `sentarse`), each holding multiple video files recorded by various participants.

- **data/processed/**  
  - `datos.csv` and `datos.xlsx`: Each row represents a frame with 33 normalized landmarks (x, y, z).  
  - `datos_preprocesados.csv`: 10-frame windows (stride = 5) with 396 extracted features (means, variances, velocities).  
  - `result_df_ventanas.csv`: Full list of generated windows, useful for analysis and debugging.

- **modelos/**  
  Stores artifacts from previous experiments:  
  - `label_encoder.pkl`: Encoder used to convert action labels into numerical indices.  
  - `modelo_movimientos_lstm.h5`: Experimental LSTM model, not used in the current pipeline.

- **notebooks/EDA.ipynb**  
  Notebook that includes:  
  1. Data exploration (PCA, t-SNE, descriptive statistics of the 396 features).  
  2. Training an `XGBClassifier` using an extensive `GridSearchCV` to evaluate multiple hyperparameter combinations and select the best model.

- **src/data_collect.py**  
  Script that iterates through `data/raw/videos/`, extracts 33 keypoints per frame using MediaPipe Pose, converts normalized coordinates to pixels, and saves the output to `data/processed/datos.csv` and `datos.xlsx`. Does not create windows or train models.

- **src/main.py**  
  Entry point for:  
  - **Train mode (`--mode train`)**:  
    1. Loads `data/processed/datos.csv`.  
    2. Creates 10-frame windows (overlap = 5) and computes 396 features per window.  
    3. Splits into train/test (70/30), applies `StandardScaler`, PCA, and trains an `XGBClassifier` via GridSearchCV.  
    4. Saves to `src/models/` the complete pipeline (`modelo_movimientos.pkl`), `label_encoder.pkl`, and optionally `scaler.pkl`.  
  - **Inference mode (`--mode inference`)**:  
    1. Loads `modelo_movimientos.pkl` and `label_encoder.pkl`.  
    2. Initializes webcam (640×480, 30 FPS). Processes every 2nd frame (~15 FPS effective).  
    3. Accumulates 10 frames in a sliding window; every 5 frames, extracts 396 features, applies scaler, PCA, and predicts with XGBoost (~220 ms per window).  
    4. Displays predicted label and probability bars on the video in the “Real-Time Activity Recognition” window (press `q` or `Esc` to exit).  
    5. (Optional) Uses a bounding-box heuristic to correct “walking forward/backward.”

- **src/models/**  
  Inference folder containing:  
  - `label_encoder.pkl`: Quick loader to map indices back to action labels.  
  - `modelo_movimientos.pkl`: Trained pipeline (scaler, PCA, and XGBoost classifier).  
  - `scaler.pkl`: Stored scaler (included inside the pipeline, exposed only if needed separately).

- **requirements.txt**  
  Lists all necessary libraries and their versions to reproduce the environment and ensure compatibility across machines.


* ![architecture.png](https://i.ibb.co/g25H9tQ/architecture.png)

The project architecture was designed modularly so that each stage (landmark extraction, data processing, model training, and real-time inference) is independent and easy to maintain. First, the `data_collect.py` script is responsible only for iterating through the raw videos and generating CSV/Excel files with the 33 keypoints per frame. Then, `main.py` combines both the training logic (PCA, scaling, and XGBoost tuning) and the live inference logic (webcam capture, 10-frame sliding window, 396-feature calculation, and prediction) into a single file. The saved models (scaler, label\_encoder, and full XGBoost pipeline) are stored in `src/models/` so that inference only needs to “load and use” them without retraining. This clear separation between raw data, preprocessing, models, and inference code allows adding improvements (e.g., changing the classifier or adjusting the window size) without affecting other parts of the system.

---

## 📋 Project Description

This system processes live video (webcam or file) to detect and classify five activities:

1. **Walking Forward**
2. **Walking Backward**
3. **Turning** (rotation in place)
4. **Standing**
5. **Sitting** (from standing to seated)

### Main Flow

1. **Landmark Extraction**

   * `src/data_collect.py` iterates through `data/raw/videos/`, extracts the 33 landmarks per frame using MediaPipe Pose, and saves them into `data/processed/datos.csv` and `datos.xlsx`.

2. **Preprocessing and Training**

   * `src/main.py` (train mode)

     * Loads `data/processed/datos.csv`
     * Generates 10-frame windows (overlap = 5) and calculates 396 features (means, variances, velocities).
     * Splits into train/test (70/30), scales with `StandardScaler`, applies PCA, and trains an `XGBClassifier` via GridSearchCV.
     * Saves `modelo_movimientos.pkl` and `label_encoder.pkl` in `src/models/`.

3. **Real-Time Inference**

   * `src/main.py` (inference mode)

     * Loads the pipeline from `src/models/modelo_movimientos.pkl` and `label_encoder.pkl`.
     * Captures frames from the webcam, processes every 2 frames (\~30 FPS), and accumulates 10 frames in a sliding queue.
     * When the queue is full and every 5 frames, extracts features, scales, applies PCA, and predicts with XGBoost (\~220 ms per window).
     * Displays the predicted label and probability percentages on the screen.

---

## 🚀 Installation and Setup

### Prerequisites

* Python 3.8 or higher
* Functional webcam (only for inference)
* GPU with CUDA (optional, to accelerate XGBoost)

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/tu_usuario/AI-Movement-Classifier.git
   cd AI-Movement-Classifier
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/macOS
   # venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify structure**

   * Ensure that `data/raw/videos/` contains folders with the videos.
   * Check that `src/models/` is initially empty (it will be populated after training).

---

## 🔄 Usage Pipeline

### A. Landmark Extraction

Run:

```bash
python src/data_collect.py
```

* Processes all videos in `data/raw/videos/`.
* Creates `data/processed/datos.csv` and `datos.xlsx` with 33 landmarks per frame.

### B. Model Training

Run:

```bash
python src/main.py --mode train
```

* Loads `data/processed/datos.csv`.
* Generates windows, calculates features, and splits into train/test.
* Fits `StandardScaler`, `PCA(n_components=50)`, and `XGBClassifier` via GridSearchCV.
* Upon completion, saves to `src/models/`:

  * `modelo_movimientos.pkl`
  * `label_encoder.pkl`
  * `scaler.pkl`

For a quick training (without GridSearch):

```bash
python src/main.py --mode train --quick
```

### C. Real-Time Inference

Run:

```bash
python src/main.py --mode inference
```

* Loads the pipeline from `src/models/`.
* Opens a video window (title “Real-Time Activity Recognition”).
* Every 10 processed frames (≈ 220 ms), displays the label and probability bars over the video.
* Press **`q`** or **`Esc`** to exit.

---

## ⚙️ Available Parameters

When running `src/main.py`, you can use:

* `--mode`:

  * `train` (training)
  * `inference` (live inference)
* `--quick` (only in `train` mode):

  * Skips GridSearchCV for a faster, rough fit
* `--n_components`: number of PCA components (default 50)
* `--n_estimators`: number of XGBoost trees (default 100)
* `--max_depth`: maximum tree depth (default 6)
* `--learning_rate`: learning rate (default 0.01)
* `--subsample` and `--colsample_bytree`: row/column fractions (default 0.8)
* `--reg_alpha`, `--reg_lambda`: L1/L2 regularization penalties (default 0 and 1)

Example usage with custom parameters:

```bash
python src/main.py --mode train \
  --n_components 100 \
  --n_estimators 200 \
  --max_depth 9 \
  --learning_rate 0.1 \
  --quick
```

---

## 🔍 Example Results

### ﹥ Evaluation

* Test set: 332 windows
* **Walking Forward**: Precision 1.00, Recall 0.96, F1 0.98
* **Walking Backward**: Precision 1.00, Recall 1.00, F1 1.00
* **Turning**: Precision 0.98, Recall 0.99, F1 0.99
* **Standing**: Precision 0.76, Recall 0.81, F1 0.78
* **Sitting**: Precision 0.79, Recall 0.76, F1 0.78
* **Macro‐avg**: P 0.91 / R 0.90 / F1 0.90

---

## 🔧 Debugging and Common Issues

1. **Camera does not open**

   * Verify that it is not in use.
   * Try a different index (`cv2.VideoCapture(1)`).

2. **Errors processing videos**

   * Make sure that the folders `caminar-adelante`, `caminar-atras`, etc., exist under `data/raw/videos/`.
   * Verify file formats: `.MOV`, `.mp4`, `.avi` are supported.

3. **Slow performance during inference**

   * Adjust `--frame_skip` in the code to process fewer frames (e.g., 3 instead of 2).
   * Reduce the camera resolution in `inference` (e.g., 320×240).

4. **Dimension error in PCA**

   * Filter out rows with NaN values in `data/processed/datos.csv` before training.
   * Verify that `result_df_ventanas.csv` contains exactly 396 feature columns.

---

## 📚 References

* T. Chen & C. Guestrin, “XGBoost: A scalable tree boosting system.” *KDD ’16*, 2016.
* C. Lugaresi et al., “MediaPipe: A framework for building perception pipelines.” *arXiv:1906.08172*, 2019.
* C. Bentéjac, A. Csörgő & G. Martínez‐Muñoz, “A comparative analysis of XGBoost.” *arXiv:1908.01161*, 2019.
* N. Yadav, “CVAT vs LabelStudio: Which one is better?” *LabelStud.io*, 2023.
* C.-H. Huang, “Real‐time motion image pose decomposition, classification and analysis.” *IJETMR*, vol. 11(6), pp. 8–13, 2024.
