## 1. `Exploration_Model.ipynb`

This notebook is used to:
1. Check GPU availability (PyTorch / TensorFlow) on Google Colab.
2. Load raw landmark data from a CSV.
3. Create sliding windows of 10 frames and extract 396 features per window.
4. Perform exploratory analysis (PCA, t-SNE, clustering).
5. Define and train an XGBoost pipeline (with PCA and `StandardScaler`) using the GPU.
6. Save the resulting artifacts (`LabelEncoder`, trained pipeline, and `StandardScaler`) to the `models/` folder.

Below are the main sections and included methods:

### 1.1 GPU Check and Mounting Google Drive

```python
import torch
import tensorflow as tf
import os

# Very important: check GPU
print("PyTorch GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch GPU name:", torch.cuda.get_device_name(0))

gpus = tf.config.list_physical_devices('GPU')
print("TensorFlow GPUs available:", gpus)

from google.colab import drive
drive.mount('/content/drive')

os.makedirs('models', exist_ok=True)
````

* **Purpose**:

  * Verify if PyTorch and TensorFlow detect a GPU (e.g., Tesla T4 on Colab).
  * Mount Google Drive to access data and save results to `models/`.

* **Method explanations**:

  * `torch.cuda.is_available()`: returns `True` if PyTorch sees a GPU.
  * `tf.config.list_physical_devices('GPU')`: lists GPU devices recognized by TensorFlow.
  * `drive.mount('/content/drive')`: mounts the Google Drive in Colab for file read/write.
  * `os.makedirs('models', exist_ok=True)`: creates the `models/` folder if it doesn’t exist (to save artifacts).

---

### 1.2 Initial Data Loading and Preprocessing

```python
import pandas as pd
import numpy as np

# Load the raw DataFrame from Drive
data = pd.read_csv('/content/drive/MyDrive/iafinal/datos.csv')

if 'frame' in data.columns:
    data = data.drop('frame', axis=1)

print("Original dimensions:", data.shape)
data.head()
```

* **Purpose**:

  * Read the CSV containing, per row, a set of 33 normalized landmarks `(x, y, z)` and its “label”.
  * Remove the `frame` column if it exists to work only with coordinates + label.

* **Method explanations**:

  * `pd.read_csv(...)`: reads a CSV file into a Pandas DataFrame.
  * `data.drop('frame', axis=1)`: drops the “frame” column if present (in some intermediate datasets).
  * `data.shape`: prints the dimensions (`n_rows, n_columns`).
  * `data.head()`: displays the first 5 rows to verify structure.

---

### 1.3 Function `create_windows_with_features`

```python
def create_windows_with_features(df_raw, window_size=10, stride=5):
    """
    Takes a DataFrame with these columns:
      - 'label'
      - 'landmark_0_x', 'landmark_0_y', 'landmark_0_z', ..., 'landmark_32_z'
    and creates windows of length `window_size` with overlap `stride`.
    For each window it extracts:
      - Mean and variance of each coordinate (x, y, z) for each of the 33 landmarks.
      - Average (vel_mean) and standard deviation (vel_std) of the differences between frames for each landmark-axis.
    Returns a new DataFrame with 396 feature columns + 1 'label' column.
    """
    n_landmarks = 33
    axes = ['x', 'y', 'z']
    windows = []

    # Iterate grouping by each label
    for label, group in df_raw.groupby('label'):
        group = group.reset_index(drop=True)
        n_frames = len(group)
        for start in range(0, n_frames - window_size + 1, stride):
            window = group.iloc[start : start + window_size].reset_index(drop=True)
            feats = {'label': label}

            # Mean and variance per landmark and axis
            for j in range(n_landmarks):
                for axis in axes:
                    col = f'landmark_{j}_{axis}'
                    vals = window[col].values
                    feats[f'mean_l{j}_{axis}'] = np.mean(vals)
                    feats[f'var_l{j}_{axis}']  = np.var(vals)

            # Velocity: differences between frames, then mean and std
            for j in range(n_landmarks):
                for axis in axes:
                    col = f'landmark_{j}_{axis}'
                    vals = window[col].values
                    diffs = np.diff(vals)
                    feats[f'vel_mean_l{j}_{axis}'] = np.mean(diffs)
                    feats[f'vel_std_l{j}_{axis}']  = np.std(diffs)

            windows.append(feats)

    df_windows = pd.DataFrame(windows)
    return df_windows
```

* **Purpose**:

  * Convert a DataFrame of individual frames into a DataFrame of “temporal windows” of size 10.
  * Each window generates 396 feature columns plus the “label” column.

* **Internal step details**:

  1. **Grouping by label**:

     * `df_raw.groupby('label')` separates rows by each action (walking-forward, walking-backward, turning, standing, sitting).
  2. **Fixed-size windows**:

     * Iterates `start` from 0 to `n_frames - window_size` in steps of `stride=5`.
     * Selects `window = group.iloc[start:start+10]`.
  3. **Mean and variance calculation**:

     * For each of the 33 landmarks, extract the column `'landmark_{j}_{axis}'` (e.g., `landmark_0_x`) and compute `np.mean(vals)` and `np.var(vals)`.
  4. **Velocity calculation**:

     * Apply `np.diff(vals)` to get the series of differences between consecutive frames (approximate instantaneous velocity).
     * Then `np.mean(diffs)` and `np.std(diffs)` for each landmark-axis.
  5. **DataFrame of windows creation**:

     * Create a dictionary `feats` with 1 + 396 entries (`label` + all features).
     * Finally, convert the list of dictionaries `windows` to `pd.DataFrame`.

---

### 1.4 Column Validation and Creation of `window_df`

```python
expected_cols = ['label'] + [f'landmark_{j}_{axis}' for j in range(33) for axis in ['x','y','z']]
missing = [c for c in expected_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

window_df = create_windows_with_features(data, window_size=10, stride=5)
print("Shape of window_df:", window_df.shape)
window_df.head()
```

* **Purpose**:

  * Ensure that the raw DataFrame (`data`) contains the 100 expected columns: `label` + 99 landmark columns (`landmark_0_x` … `landmark_32_z`).
  * If any are missing, raise an explicit error.
  * Then generate `window_df` with rows = windows and columns = 397 (1 `label` column + 396 features).

* **Method explanations**:

  * `expected_cols`: list of expected column names.
  * The condition `if missing:` ensures no expected columns are missing before creating windows.

---

### 1.5 Label Encoding and Train/Test Split

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# LabelEncoder to convert text labels to numeric indices
le = LabelEncoder()
y_encoded = le.fit_transform(window_df['label'])

X = window_df.drop(columns=['label'])
y = y_encoded

# Stratified split: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Print class distributions
import numpy as np
for split_name, labels in zip(['Train', 'Test'], [y_train, y_test]):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"{split_name} distribution:")
    for u, c in zip(unique, counts):
        print(f"  {le.inverse_transform([u])[0]}: {c} samples")
```

* **Purpose**:

  * Convert the `label` column (strings) to numeric vectors (`0,1,2,3,4`) using `LabelEncoder`.
  * Separate features (`X`) from labels (`y`).
  * Split into training set (`X_train`, `y_train`) and test set (`X_test`, `y_test`) in a stratified way to preserve class proportions.

* **Method explanations**:

  * `LabelEncoder().fit_transform(...)`: assigns a unique index to each action label.
  * `train_test_split(..., stratify=y)`: preserves the same class proportion in train and test sets.
  * `np.unique(labels, return_counts=True)`: counts how many samples of each class are in each split.

---

### 1.6 Principal Component Analysis (PCA) on the Entire Dataset

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Normalize all X (train+test together)
scaler_all = StandardScaler()
X_scaled_all = scaler_all.fit_transform(X)

# Fit PCA without specifying number of components
pca_full = PCA()
pca_full.fit(X_scaled_all)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("Explained variance by each component (only > 0.001):")
for i, var_exp in enumerate(explained_variance_ratio):
    if var_exp > 0.001:
        print(f"Component {i+1}: {var_exp:.4f}")

# Plot of cumulative explained variance
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance,
         marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
```

* **Purpose**:

  * Understand how many principal components explain most of the variance of the 396 features.
  * Choose a suitable `n_components` value (e.g., 50) when training the final model.

* **Method explanations**:

  * `StandardScaler().fit_transform(X)`: normalizes each feature to mean 0 and variance 1.
  * `PCA().fit(...)`: fits PCA to the entire normalized matrix.
  * `pca_full.explained_variance_ratio_`: array with the proportion of variance explained by each component.
  * `np.cumsum(...)`: cumulative sum to plot total explained variance.
  * We filter to only print components with `var_exp > 0.001` (0.1% of variance).
  * The plot “components vs. cumulative explained variance” helps decide how many to use.

---

### 1.7 2D Visualization: PCA and t-SNE

#### 1.7.1 PCA 2D

```python
# Reduce to 2 components for plotting
pca_2 = PCA(n_components=2)
X_reduced_pca2 = pca_2.fit_transform(X_scaled_all)

# Assign unique colors per label
unique_labels = np.unique(y)
label_colors = {lbl: plt.get_cmap('tab10')(i) for i, lbl in enumerate(unique_labels)}
colors = [label_colors[val] for val in y]

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_pca2[:, 0], X_reduced_pca2[:, 1], c=colors, s=15)
plt.title("2D Distribution (PCA) of Windows")
for lbl, col in label_colors.items():
    plt.scatter([], [], color=col, label=le.inverse_transform([lbl])[0])
plt.legend()
plt.show()
```

* **Purpose**:

  * See how windows are spread in the space reduced to 2 dimensions via PCA, colored by their original label.

* **Method explanations**:

  * `PCA(n_components=2).fit_transform(...)`: projects normalized data into 2 dimensions.
  * `plt.scatter(...)`: plots points in 2D, colored by label.
  * The legend (`plt.legend()`) shows which color corresponds to each action class.

#### 1.7.2 t-SNE 2D

```python
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init='pca',
    random_state=42
)
X_tsne2 = tsne.fit_transform(X_scaled_all)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=colors, s=15)
plt.title("2D Distribution (t-SNE) of Windows")
for lbl, col in label_colors.items():
    plt.scatter([], [], color=col, label=le.inverse_transform([lbl])[0])
plt.legend()
plt.show()
```

* **Purpose**:

  * Visualize the non-linear projection of data into 2D with t-SNE to check class separation or overlaps.

* **Method explanations**:

  * `TSNE(...)`: sets typical parameters (`perplexity`, `learning_rate`, `init='pca'`).
  * `fit_transform(X_scaled_all)`: applies t-SNE to the entire normalized dataset.
  * Plot is similar to PCA but clusters may appear more defined.

---

### 1.8 Clustering with KMeans on PCA-2D

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(X_reduced_pca2)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced_pca2[:, 0], X_reduced_pca2[:, 1], c=cluster_labels, cmap='viridis', s=15)
plt.title("KMeans (5 clusters) on PCA-2D")
plt.show()
```

* **Purpose**:

  * Evaluate if a clustering algorithm (KMeans with 5 clusters) matches the activity classification.
  * Visualize, in the same PCA-2D space, the groups found by KMeans.

* **Method explanations**:

  * `KMeans(n_clusters=5, random_state=0).fit_predict(...)`: fits the model and returns an array of cluster labels for each point.
  * `plt.scatter(..., c=cluster_labels, cmap='viridis')`: colors points according to their assigned cluster.

---

### 1.9 Defining the XGBoost Pipeline and GridSearchCV

```python
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, ParameterGrid
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib

# Pipeline that includes: StandardScaler → PCA → XGBoost (configured for GPU)
pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        random_state=42,
        verbosity=0
    ))
])

# Hyperparameter grid definition
param_grid_xgb = {
    'pca__n_components':     [50, 100],
    'clf__n_estimators':     [100, 200],
    'clf__max_depth':        [6, 9],
    'clf__learning_rate':    [0.01, 0.1],
    'clf__subsample':        [0.8, 1],
    'clf__colsample_bytree': [0.8, 1],
    'clf__reg_alpha':        [0, 0.5],
    'clf__reg_lambda':       [1, 2]
}

# GridSearchCV setup (3 folds, scoring='accuracy')
grid_xgb = GridSearchCV(
    estimator=pipeline_xgb,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='accuracy',
    verbose=0,
    n_jobs=-1
)

# Total number of combinations
from sklearn.model_selection import ParameterGrid
total_combos = len(list(ParameterGrid(param_grid_xgb)))

# Progress bar with tqdm-joblib
with tqdm_joblib(tqdm(desc="GridSearch Progress", total=total_combos)):
    grid_xgb.fit(X_train, y_train)

print("Best XGBoost parameters (GPU):")
print(grid_xgb.best_params_)
best_xgb_model = grid_xgb.best_estimator_
```

* **Purpose**:

  * Train and select the best XGBoost pipeline on GPU, chaining `StandardScaler`, `PCA(n_components)`, and `XGBClassifier(tree_method='gpu_hist')`.
  * Use `GridSearchCV` to explore 256 hyperparameter combinations with 3-fold cross-validation, optimizing “accuracy”.

* **Method explanations**:

  * `Pipeline([('scaler', ...), ('pca', ...), ('clf', ...)])`: chains preprocessing and model into one object.
  * `XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')`: forces XGBoost to use GPU for faster training.
  * `GridSearchCV(estimator, param_grid, cv=3, scoring='accuracy', n_jobs=-1)`:

    * Tries all hyperparameter combinations defined in `param_grid_xgb`.
    * `cv=3`: three internal validation folds.
    * `n_jobs=-1`: uses all available CPU cores.
  * `tqdm_joblib(...)`: wrapper to show a progress bar in Colab while running GridSearch.
  * `grid_xgb.best_params_`: prints the best parameter set found.
  * `best_xgb_model = grid_xgb.best_estimator_`: pipeline object with the best parameters.

---

### 1.10 Test Set Evaluation and Model Serialization

```python
from sklearn.metrics import accuracy_score, classification_report

# Extract substeps from the trained pipeline
scaler = best_xgb_model.named_steps['scaler']
pca    = best_xgb_model.named_steps['pca']
clf    = best_xgb_model.named_steps['clf']

# Preprocess X_test
X_test_scaled = scaler.transform(X_test)
X_test_pca    = pca.transform(X_test_scaled)

# Predictions and metrics
y_pred_xgb = clf.predict(X_test_pca)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost test accuracy: {acc_xgb:.4f}\n")

print("XGBoost Classification Report:")
print(classification_report(
    y_test,
    y_pred_xgb,
    target_names=le.inverse_transform(unique_labels)
))

# Save artifacts to disk (models/)
import joblib
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(best_xgb_model, 'models/movement_model.pkl')
scaler = best_xgb_model.named_steps['scaler']
joblib.dump(scaler, 'models/scaler.pkl')

print("LabelEncoder saved at: models/label_encoder.pkl")
print("Pipeline (scaler + PCA + XGBoost) saved at: models/movement_model.pkl")
print("Scaler saved at: models/scaler.pkl")
```

* **Purpose**:

  1. Evaluate the performance of the best pipeline on the test set (`X_test`, `y_test`).
  2. Print `accuracy` and a detailed `classification_report` (precision, recall, f1-score, support) for each action label.
  3. Serialize (save) the following files in `models/`:

     * `label_encoder.pkl`
     * `movement_model.pkl` (full pipeline)
     * `scaler.pkl` (saving only the `StandardScaler` object)

* **Method explanations**:

  * `scaler.transform(X_test)`: applies normalization based on training.
  * `pca.transform(...)`: projects standardized data into the reduced space.
  * `clf.predict(...)`: predicts labels in the PCA space.
  * `accuracy_score(y_test, y_pred_xgb)`: computes overall accuracy.
  * `classification_report(...)`: shows metrics per class.
  * `joblib.dump(obj, path)`: serializes the Python object to disk (optimized pickle).

---

## 2. `main.py`

This script is the entry point for **real-time inference** using the webcam. Below is a detailed explanation of each block, function, and method.

### 2.1 Library Imports and Initial Definitions

```python
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import sys
```

* **cv2**: OpenCV for capturing and displaying real-time video.
* **mediapipe**: Google’s library for pose detection and landmark extraction.
* **numpy**: numerical computations.
* **joblib**: load serialized objects (`.pkl`).
* **collections.deque**: fixed-size queue to store windows of landmarks.
* **sys**: exit the program in case of error.

---

### 2.2 Function `features_from_window(frames_list)`

```python
def features_from_window(frames_list):
    """
    Receives a list of `window_size` arrays of shape (99,) in pixels, each:
      [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32]
    Returns a vector of 396 features:
      - For each of the 33 landmarks (indices 0 to 32) and each dimension (x, y, z):
          * mean of the values in the window
          * variance of the values in the window
      - For each landmark and dimension (x, y, z):
          * mean of the velocities (differences between frames)
          * standard deviation of the velocities
    """
    window_size = len(frames_list)
    n_landmarks = 33
    # Stack 'window_size' arrays of shape (99,) into an array of shape (window_size, 33, 3)
    arr = np.stack(frames_list, axis=0).reshape(window_size, n_landmarks, 3)

    feats = []

    # Mean and variance per landmark and coordinate
    for j in range(n_landmarks):
        for a in range(3):
            vals = arr[:, j, a]      # axis a in landmark j across all frames
            feats.append(np.mean(vals))
            feats.append(np.var(vals))

    # Compute differences (velocity) between consecutive frames
    diffs = np.diff(arr, axis=0)  # shape = (window_size-1, 33, 3)
    for j in range(n_landmarks):
        for a in range(3):
            vel = diffs[:, j, a]    # velocities over the window
            feats.append(np.mean(vel))
            feats.append(np.std(vel))

    return np.array(feats)  # 396-length vector
```

* **Purpose**:

  * Given a `frames_list` of length 10, where each element is a 99-element array (33 landmarks × 3 coordinates), calculate statistics (mean, variance, mean velocity, and velocity standard deviation) for all landmark-coordinate combinations.

* **Step explanations**:

  1. `np.stack(frames_list, axis=0)`: stacks the list of `(99,)` vectors into a matrix `(window_size, 99)`.
  2. `.reshape(window_size, 33, 3)`: reshapes to `(n_frames, n_landmarks, n_dimensions)`.
  3. First loop: for each `j` in `0..32` (landmark) and `a` in `0..2` (x,y,z):

     * `vals = arr[:, j, a]`: extract values for that coordinate over the window.
     * `np.mean(vals)` and `np.var(vals)`: compute mean and variance.
  4. `diffs = np.diff(arr, axis=0)`: compute differences in each coordinate between consecutive frames (approximate velocity).
  5. Second loop: for each landmark and coordinate, extract `vel = diffs[:, j, a]` and compute `np.mean(vel)` and `np.std(vel)`.

---

### 2.3 Function `extract_landmarks(results)`

```python
def extract_landmarks(results):
    """
    Given the MediaPipe Pose result object (`results`),
    extract the 33 normalized landmarks (x, y, z) if present.
    Returns an array of shape (99,) with [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32].
    If no pose is detected (`results.pose_landmarks == None`), returns None.
    """
    if results.pose_landmarks:
        return np.array([
            [lmk.x, lmk.y, lmk.z] 
            for lmk in results.pose_landmarks.landmark
        ]).flatten()
    return None
```

* **Purpose**:

  * Convert the MediaPipe output (`results.pose_landmarks.landmark`) to a 1D vector of length 99 with normalized coordinates.

* **Method explanations**:

  * `results.pose_landmarks.landmark`: list of 33 Landmark objects, each with attributes `x`, `y`, `z` in range \[0,1].
  * `[ [lmk.x, lmk.y, lmk.z] for lmk in ... ]`: list of 33 sublists `[x, y, z]`.
  * `.flatten()`: flattens the `(33,3)` array into a `(99,)` vector.

---

### 2.4 Function `bbox_height(frame_lmks)`

```python
def bbox_height(frame_lmks):
    """
    Calculates the height in pixels of the body in a frame:
      - Receives `frame_lmks`: a 99-element array [x0, y0, z0, ..., x32, y32, z32]
      - Extracts the Y values of each landmark: indices 1, 4, 7, ..., 97
      - Returns difference: (y_max - y_min)
    """
    ys = frame_lmks[1::3]  # every third element, starting at index 1 (Y coordinates)
    return max(ys) - min(ys)
```

* **Purpose**:

  * Given a `(99,)` vector of pixel coordinates, calculate the bounding-box height (vertical height of the body) as `y_max − y_min`.

* **Method explanations**:

  * `frame_lmks[1::3]`: every third element in the array, starting at index 1, corresponds to the Y coordinates of the 33 landmarks.
  * `max(ys) − min(ys)`: vertical distance in pixels.

---

### 2.5 Function `estimate_delta_bbox(frames_list)`

```python
def estimate_delta_bbox(frames_list):
    """
    Measures the relative change in bounding-box height from the first frame
    to the last frame of the window:
      delta_h = (h_last - h_first) / h_first
    If h_first == 0, returns 0.0 to avoid division by zero.
    """
    h0 = bbox_height(frames_list[0])
    hN = bbox_height(frames_list[-1])
    if h0 == 0:
        return 0.0
    return (hN - h0) / h0
```

* **Purpose**:

  * Calculate how the “torso” height (bounding box) changes between the first and last frame of the window.
  * Use this value (`delta_h`) to infer vertical displacement (e.g., stepping forward vs. backward) if the classifier’s probability is low.

* **Step explanations**:

  * `h0 = bbox_height(frames_list[0])`: initial height.
  * `hN = bbox_height(frames_list[-1])`: height in the final frame.
  * `(hN - h0) / h0`: relative change percentage.
  * If `h0 == 0`, avoid division by zero and return `0.0`.

---

### 2.6 Function `create_opencv_window()`

```python
def create_opencv_window():
    global window_created
    if not window_created:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_created = True
```

* **Purpose**:

  * Create an OpenCV window (only once) with the name `window_name`.
  * Controlled by the global boolean `window_created` to avoid recreating the window at every iteration.

* **Method explanations**:

  * `cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)`: creates the window and automatically resizes it to fit the content.

---

### 2.7 Function `cleanup()`

```python
def cleanup():
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
```

* **Purpose**:

  * Release resources upon completion or interruption:

    1. `cap.release()`: closes the camera.
    2. `cv2.destroyAllWindows()`: closes all OpenCV windows.
    3. `pose.close()`: closes the MediaPipe Pose object.

* **Method explanations**:

  * Always called in the `finally` block to ensure resources are freed even if an error occurs or the user presses Ctrl+C.

---

### 2.8 Loading Models and Existence Validation

```python
try:
    model = joblib.load('models/movement_model.pkl')
except FileNotFoundError:
    print("❌ Error: movement_model.pkl not found")
    sys.exit(1)

try:
    le = joblib.load('models/label_encoder.pkl')
except FileNotFoundError:
    print("❌ Error: label_encoder.pkl not found")
    sys.exit(1)

try:
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded successfully")
except FileNotFoundError:
    print("❌ Error: scaler.pkl not found")
    print("📝 You need to save the scaler during training")
    sys.exit(1)

print("✅ Model and LabelEncoder loaded")
```

* **Purpose**:

  * Attempt to load the serialized files from `models/`.
  * If any file is missing, print a clear error and exit the program (`sys.exit(1)`).

* **Method explanations**:

  * `joblib.load(path)`: loads an object previously serialized with `joblib.dump`.
  * Catches `FileNotFoundError` for each critical file and displays a specific error message.

---

### 2.9 Extracting Substeps from the Trained Pipeline

```python
pca_trained = model.named_steps['pca']
clf_trained = model.named_steps['clf']
```

* **Purpose**:

  * Get direct references to the `pca` and `clf` (XGBoost) stages of the pipeline loaded as `model`.
  * Allows applying transformations and predictions manually (without calling `model.predict(...)` directly).

* **Method explanations**:

  * `model.named_steps['pca']`: the fitted `PCA(n_components=50)` object from training.
  * `model.named_steps['clf']`: the trained `XGBClassifier` object.

---

### 2.10 Initializing MediaPipe Pose and Configuring the Camera

```python
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
    print("❌ Could not access the camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

* **Purpose**:

  1. Configure the MediaPipe Pose module with appropriate parameters for real-time video:

     * `static_image_mode=False`: optimizes for continuous video.
     * `model_complexity=1`: medium complexity model.
     * `smooth_landmarks=True`: smooths small variations between frames.
     * `min_detection_confidence=0.5` and `min_tracking_confidence=0.5`: minimum thresholds for detection and tracking.
  2. Initialize the webcam with OpenCV.
  3. Set resolution to 640×480, 30 FPS, and minimal buffer.

* **Method explanations**:

  * `mp_pose.Pose(...)`: creates a Pose object to process frames.
  * `cv2.VideoCapture(0)`: opens the camera at index 0 by default.
  * `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`, etc.: set capture properties.
  * `if not cap.isOpened()`: checks if the camera is available; if not, exits the program.

---

### 2.11 Global Variables for Window and Frame Queue

```python
window_size = 10
frames_queue = deque(maxlen=window_size)

frame_skip = 2
frame_count = 0

window_name = 'Movement Prediction'
window_created = False
```

* **Purpose**:

  * `window_size = 10`: fixed length of the frame window.
  * `frames_queue = deque(maxlen=10)`: FIFO queue that stores up to 10 arrays of 99 values (landmark pixels). When full and a new item is appended, the oldest is discarded.
  * `frame_skip = 2`: process only 1 of every 2 frames (to reduce load).
  * `frame_count = 0`: sequential frame counter to apply skipping.
  * `window_name = 'Movement Prediction'`: name assigned to the OpenCV window.
  * `window_created = False`: flag to control the creation of the window (only once).

---

### 2.12 Main Capture and Prediction Loop

```python
try:
    print("🎥 Starting video capture. Press 'q' to exit.")
    create_opencv_window()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame from camera")
            break

        # Skip frames according to frame_skip
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
            )

        # Extract normalized landmarks
        landmarks_norm = extract_landmarks(results)
        if landmarks_norm is not None:
            # Convert normalized coordinates to pixels
            landmarks_px = landmarks_norm.copy()
            for i in range(33):
                landmarks_px[i*3 + 0] = landmarks_norm[i*3 + 0] * w   # x * width
                landmarks_px[i*3 + 1] = landmarks_norm[i*3 + 1] * h   # y * height

            # Append to the frame queue (shape (99,))
            frames_queue.append(landmarks_px)

            # If the queue has 10 frames and frame_count % 5 == 0 → predict
            if len(frames_queue) == window_size and frame_count % 5 == 0:
                try:
                    # 1. Extract features from the window
                    feats_window = features_from_window(list(frames_queue))  # (396,)

                    # 2. Scale and project with PCA
                    X_scaled = scaler.transform(feats_window.reshape(1, -1))
                    X_pca = pca_trained.transform(X_scaled)

                    # 3. Predict probabilities with XGBoost
                    proba = clf_trained.predict_proba(X_pca)[0]
                    idx_pred = np.argmax(proba)
                    label = le.inverse_transform([idx_pred])[0]

                    # 4. Heuristic based on height change (delta_h)
                    delta_h = estimate_delta_bbox(list(frames_queue))
                    height_threshold = 0.08
                    override = None
                    if delta_h > height_threshold:
                        override = 'walking-forward'
                    elif delta_h < -height_threshold:
                        override = 'walking-backward'

                    # If probability is low (< 0.70) and there is an override, use the override
                    if override is not None and proba[idx_pred] < 0.70:
                        label = override

                    # 5. Draw a black rectangle background for text
                    cv2.rectangle(frame, (5, 5), (500, 140), (0, 0, 0), -1)
                    # 6. Write the main label
                    cv2.putText(frame, f'Movement: {label}', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 7. Write probabilities for each class
                    for i, cls in enumerate(le.classes_):
                        cv2.putText(frame, f'{cls}: {proba[i]*100:4.1f}%', (10, 65 + 20*i),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                except Exception as e:
                    # Handle prediction errors
                    print(f"⚠️ Error during prediction: {e}")
                    cv2.rectangle(frame, (5, 5), (350, 50), (0, 0, 0), -1)
                    cv2.putText(frame, 'Prediction error', (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Permanent exit message
        cv2.putText(frame, "Press 'q' to exit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the annotated frame
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("🛑 Exit requested by user.")
            break
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 Window closed by user.")
            break

except KeyboardInterrupt:
    print("🛑 Manual interruption detected (Ctrl+C).")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
finally:
    cleanup()
    print("✅ Program terminated successfully.")
```

#### Step-by-Step Breakdown

1. **Start of the `try` block**

   * Prints an initial message: “Starting video capture. Press 'q' to exit.”
   * Calls `create_opencv_window()` to ensure the OpenCV window exists.

2. **`while True` loop (continuous capture)**

   * **Reading a frame**: `ret, frame = cap.read()`

     * If `ret == False`, the frame couldn’t be read → break the loop.
   * **Frame skipping**:

     * If `frame_count % frame_skip != 0`, skip this frame (increment `frame_count` and `continue`).
     * This effectively processes 1 out of every 2 (or every N) frames, reducing computational load.
     * After passing the filter, `frame_count` is incremented.

3. **Pose processing on the frame**

   * `h, w, _ = frame.shape`: height and width of the current frame.
   * `rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`: convert BGR → RGB for MediaPipe.
   * `rgb_frame.flags.writeable = False`: optimization to prevent memory copying.
   * `results = pose.process(rgb_frame)`: MediaPipe processes the frame and returns an object with landmarks.
   * `rgb_frame.flags.writeable = True`: revert mutability.

4. **Drawing landmarks on the frame**

   * If `results.pose_landmarks` exists, call:

     ```python
     mp_drawing.draw_landmarks(
         frame,
         results.pose_landmarks,
         mp_pose.POSE_CONNECTIONS,
         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
     )
     ```
   * Draws lines between landmarks and circles at each key point with specific colors.

5. **Landmark extraction and conversion to pixels**

   * `landmarks_norm = extract_landmarks(results)`: returns a `(99,)` array with normalized values in \[0,1].
   * If not `None`, convert to pixel coordinates:

     ```python
     landmarks_px = landmarks_norm.copy()
     for i in range(33):
         landmarks_px[i*3 + 0] = landmarks_norm[i*3 + 0] * w   # X in pixels
         landmarks_px[i*3 + 1] = landmarks_norm[i*3 + 1] * h   # Y in pixels
     ```
   * The resulting `landmarks_px` array is length 99, with pixel coordinates matching the camera resolution.

6. **Enqueue the landmark vector in `frames_queue`**

   * `frames_queue.append(landmarks_px)`
   * If `frames_queue` already has 10 elements, the oldest is automatically discarded.

7. **Prediction every 5 iterations (when the queue is full)**

   * Check: `if len(frames_queue) == window_size and frame_count % 5 == 0:`

   * Then:

     1. **Feature extraction**:

        ```python
        feats_window = features_from_window(list(frames_queue))  # (396,)
        ```

     2. **Scaling and PCA**:

        ```python
        X_scaled = scaler.transform(feats_window.reshape(1, -1))
        X_pca = pca_trained.transform(X_scaled)
        ```

     3. **Prediction with XGBoost**:

        ```python
        proba = clf_trained.predict_proba(X_pca)[0]
        idx_pred = np.argmax(proba)
        label = le.inverse_transform([idx_pred])[0]
        ```

     4. **Height change heuristic**:

        ```python
        delta_h = estimate_delta_bbox(list(frames_queue))
        height_threshold = 0.08
        override = None
        if delta_h > height_threshold:
            override = 'walking-forward'
        elif delta_h < -height_threshold:
            override = 'walking-backward'
        if override is not None and proba[idx_pred] < 0.70:
            label = override
        ```

        * If the relative height change is greater than 8% → “walking-forward”.
        * If less than −8% → “walking-backward”.
        * Only apply if the classifier’s confidence is below 70%.

     5. **Draw rectangle and result text**:

        ```python
        cv2.rectangle(frame, (5, 5), (500, 140), (0, 0, 0), -1)
        cv2.putText(frame, f'Movement: {label}', (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        for i, cls in enumerate(le.classes_):
            cv2.putText(frame, f'{cls}: {proba[i]*100:4.1f}%', 
                        (10, 65 + 20*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        ```

        * The black rectangle background improves text readability.
        * Prints the main label in green and each class’s probability in light gray.

   * **Error handling in prediction**:
     If there is an error (e.g., incorrect dimensions), it is caught by `except Exception as e:` and prints:

     ```python
     print(f"⚠️ Error during prediction: {e}")
     cv2.rectangle(frame, (5, 5), (350, 50), (0, 0, 0), -1)
     cv2.putText(frame, 'Prediction error', (10, 35),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
     ```

8. **Permanent exit message**

   ```python
   cv2.putText(frame, "Press 'q' to exit", 
               (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
   ```

   * Text at the bottom reminding how to exit.

9. **Display the frame on screen**

   ```python
   cv2.imshow(window_name, frame)
   ```

   * Updates the “Movement Prediction” window with the annotated frame.

10. **Key capture**

    ```python
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("🛑 Exit requested by user.")
        break
    elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("🛑 Window closed by user.")
        break
    ```

    * If `'q'` (ASCII `ord('q')`) or Esc (`27`) is pressed, break the loop.
    * If the user closes the window manually, `getWindowProperty(...) < 1` detects the window no longer exists and breaks the loop.

11. **`except` and `finally` blocks**

    ```python
    except KeyboardInterrupt:
        print("🛑 Manual interruption detected (Ctrl+C).")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        cleanup()
        print("✅ Program terminated successfully.")
    ```

    * **`KeyboardInterrupt`**: if the user presses Ctrl+C in the terminal, it’s caught and a message is printed.
    * **`Exception as e`**: captures any other unforeseen error and displays it.
    * **`finally`**: ensures `cleanup()` is called to release the camera and close windows.

---

## 3. Summary of Functions and Methods

| Function / Method                        | Location                                | Purpose                                                                                                                            |
| ---------------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `torch.cuda.is_available()`              | `Exploration_Model.ipynb`               | Check if PyTorch recognizes a GPU.                                                                                                 |
| `tf.config.list_physical_devices('GPU')` | `Exploration_Model.ipynb`               | Check which GPUs TensorFlow recognizes.                                                                                            |
| `drive.mount(...)`                       | `Exploration_Model.ipynb`               | Mount Google Drive in Colab for data access and saving.                                                                            |
| `create_windows_with_features(...)`      | `Exploration_Model.ipynb`               | Generate windows of length 10 with overlap 5 and extract 396 features (mean, var, vel\_mean, vel\_std) for each landmark.          |
| `LabelEncoder().fit_transform()`         | `Exploration_Model.ipynb`               | Convert text labels to numeric indices (0–4).                                                                                      |
| `train_test_split(..., stratify=y)`      | `Exploration_Model.ipynb`               | Split data into train and test sets while maintaining class proportions.                                                           |
| `StandardScaler().fit_transform()`       | `Exploration_Model.ipynb`               | Normalize all features for PCA analysis.                                                                                           |
| `PCA().fit(...)`                         | `Exploration_Model.ipynb`               | Fit PCA without specifying number of components (to calculate explained variance).                                                 |
| `PCA(n_components=2).fit_transform(...)` | `Exploration_Model.ipynb`               | Reduce data to 2 dimensions for visualization (scatter plot).                                                                      |
| `TSNE(...).fit_transform(...)`           | `Exploration_Model.ipynb`               | Compute non-linear t-SNE projection in 2D for visualization.                                                                       |
| `KMeans(n_clusters=5).fit_predict(...)`  | `Exploration_Model.ipynb`               | Apply KMeans on PCA-2D embedding to compare clustering with actual labels.                                                         |
| `Pipeline([...])`                        | `Exploration_Model.ipynb`               | Chain `StandardScaler`, `PCA`, and `XGBClassifier` into one object.                                                                |
| `GridSearchCV(...)`                      | `Exploration_Model.ipynb`               | Search for the best hyperparameter combination for XGBoost on GPU.                                                                 |
| `joblib.dump(...)`                       | `Exploration_Model.ipynb` and `main.py` | Serialize Python objects (`LabelEncoder`, pipeline, `StandardScaler`) to `.pkl` files.                                             |
| `joblib.load(...)`                       | `main.py`                               | Load the `.pkl` files containing the trained pipeline, label encoder, and scaler.                                                  |
| `mp_pose.Pose(...)`                      | `main.py`                               | Create the MediaPipe Pose object for real-time landmark detection and tracking.                                                    |
| `cv2.VideoCapture(0)`                    | `main.py`                               | Initialize the webcam for real-time video capture.                                                                                 |
| `extract_landmarks(results)`             | `main.py`                               | Convert MediaPipe output (normalized landmarks) into a `(99,)` array.                                                              |
| `bbox_height(frame_lmks)`                | `main.py`                               | Calculate the vertical height of the bounding box (maxY−minY) from a `(99,)` landmarks vector in pixels.                           |
| `estimate_delta_bbox(frames_list)`       | `main.py`                               | Calculate `(h_last − h_first) / h_first` to detect relative height change between the first and last frame of the window.          |
| `features_from_window(frames_list)`      | `main.py`                               | Extract 396 features from a 10-frame window: mean, variance, mean velocity, and velocity standard deviation.                       |
| `create_opencv_window()`                 | `main.py`                               | Create (only once) the OpenCV window to display real-time predictions.                                                             |
| `cleanup()`                              | `main.py`                               | Release the camera (`cap.release()`), close OpenCV windows (`cv2.destroyAllWindows()`), and close MediaPipe Pose (`pose.close()`). |

---

## 4. Complete Execution Flow

1. **Exploration and Training**

   * In **`Exploration_Model.ipynb`**, the steps are:

     1. Check GPU and mount Drive.
     2. Load raw landmark data (`datos.csv`).
     3. Create windows of 10 frames with `create_windows_with_features(...)`.
     4. Encode labels and split into train/test.
     5. Perform PCA analysis to choose `n_components`.
     6. Visualize embeddings with PCA-2D and t-SNE.
     7. (Optional) Apply KMeans on PCA-2D.
     8. Define and train an XGBoost pipeline on GPU with `GridSearchCV`.
     9. Evaluate metrics on the test set.
     10. Save `label_encoder.pkl`, `movement_model.pkl`, and `scaler.pkl` in `models/`.

2. **Real-Time Inference**

   * When running **`main.py`**:

     1. Load `movement_model.pkl`, `label_encoder.pkl`, and `scaler.pkl`.
     2. Initialize MediaPipe Pose and the camera (640×480, 30 FPS).
     3. Create a `frames_queue` of length 10 and start a loop to read frames continuously.
     4. Every 2 frames (due to `frame_skip = 2`), process the frame:

        * Convert normalized landmarks to pixels and enqueue them.
        * Every 5 iterations, if there are 10 frames in the queue, extract 396 features with `features_from_window(...)`.
        * Normalize (`scaler.transform`), apply PCA, and predict with XGBoost.
        * Use the `delta_h` heuristic to adjust “walking-forward” vs. “walking-backward” if confidence is low.
        * Draw the label, probabilities, and an exit message on the video.
     5. Display the “Movement Prediction” window with `cv2.imshow(...)`.
     6. If the user presses `'q'` or closes the window, exit the loop.
     7. In the `finally` block, call `cleanup()` to release all resources.

---

## 5. Conclusion

This document provides a comprehensive overview of:

* **Exploration Notebook (`Exploration_Model.ipynb`)**: data loading, window and feature creation, PCA/t-SNE analysis, clustering, GPU-trained XGBoost pipeline with hyperparameter optimization, and model serialization.
* **Inference Script (`main.py`)**: loading trained artifacts, real-time landmark extraction, building 10-frame windows, extracting 396 features, scaling, PCA, predicting with XGBoost, bounding-box heuristic, and visualizing results on screen.

Each function and method is described in detail to understand its purpose and operation within the full human activity classification pipeline. With this documentation you can:

1. Reproduce training in Colab or locally.
2. Adjust PCA parameters, XGBoost hyperparameters, heuristic thresholds (`height_threshold`), etc.
3. Extend the project by adding new actions or changing the model architecture.
