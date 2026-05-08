# SmartPresence: Working Science and System Explanation

## 1) What problem this solves

SmartPresence automates attendance by recognizing known people from a webcam stream and writing attendance records to CSV.  
It is designed as a **real-time closed-set identification** system with **open-set rejection**:

- Closed-set: classify among known enrolled identities (your 3 teammates).
- Open-set rejection: if confidence is low, output **Unknown** instead of forcing a wrong identity.

This reduces manual roll-call overhead and limits proxy attendance risk.

---

## 2) High-level scientific idea

The system combines two stages:

1. **Perception stage (fixed pretrained vision model):**
   - Detect faces.
   - Align faces to a canonical geometry.
   - Convert each face into a compact numeric embedding (feature vector).

2. **Decision stage (small trainable classifier):**
   - Learn boundaries between team members in embedding space.
   - Output class probabilities.
   - Apply threshold $\tau$ for Unknown rejection.

This is standard **transfer learning**: reuse a strong pretrained encoder, train only a small head on local data.

---

## 3) End-to-end pipeline (what runs in order)

1. `src/capture_dataset.py`  
   Collect raw webcam images per person.

2. `src/preprocess.py`  
   Detect single face, align to 5-point template, resize to 160x160, augment.

3. `src/encode.py`  
   Convert processed images to embeddings (`X.npy`, `y.npy`).

4. `src/train_classifier.py`  
   Train and select best classifier head (SVM/MLP) via validation macro-F1.

5. `src/evaluate.py`  
   Produce confusion matrix, per-class metrics, t-SNE, open-set threshold curve.

6. `src/recognize.py`  
   Live inference + attendance logging.

7. `scripts/make_diagrams.py`  
   Generate report figures.

---

## 4) Data and labels

### Input data type
- RGB face images from webcam.
- Organized by identity label.

### Labels
- Class IDs for known identities:
  - `ahsan_22K-4176`
  - `anas_m_22K-4548`
  - `anas_k_22K-4483`

### Why multiple images per person
A face is not one static pattern; appearance changes with:
- pose,
- expression,
- illumination,
- camera distance/focus.

Multiple samples help classifier boundaries generalize to live conditions.

---

## 5) Preprocessing science

Implemented in `src/preprocess.py`.

### 5.1 Face detection
`face_recognition.face_locations(..., model="hog")`

HOG features capture gradient orientation structure and feed a detector learned on face-like patterns. It is CPU-friendly but weaker than modern CNN detectors for extreme poses.

### 5.2 Landmark-based alignment
Landmarks are reduced to 5 key points:
- left eye center,
- right eye center,
- nose tip,
- left mouth corner,
- right mouth corner.

Alignment estimates a similarity transform:

$$
T^*=\arg\min_T \sum_{i=1}^{5}\|T(p_i)-q_i\|_2^2
$$

where:
- $p_i$ = detected points,
- $q_i$ = canonical ArcFace anchor points.

This step reduces nuisance variation (rotation/translation/scale), so identity information becomes easier to model.

### 5.3 Augmentation
- Horizontal flip
- Rotation in $\pm 15^\circ$
- Brightness scaling
- Gaussian noise

This approximates real deployment perturbations and reduces overfitting.

---

## 6) Embedding science (feature extraction)

Implemented in `src/encode.py`.

### 6.1 Encoder options
- `face_recognition` backend -> 128-d embedding (dlib ResNet-based model)
- `facenet_pytorch` backend -> 512-d embedding (`InceptionResnetV1`)

### 6.2 Why embeddings
Raw pixels are high-dimensional and unstable. Embeddings map faces into a metric space where:
- same identity -> closer vectors,
- different identities -> farther vectors.

### 6.3 L2 normalization
Each vector is normalized to unit norm:

$$
\tilde{x} = \frac{x}{\|x\|_2}
$$

This stabilizes scale and makes distance/decision geometry more consistent.

---

## 7) Classifier-head science

Implemented in `src/train_classifier.py`.

### 7.1 Why train only the head
You have a small local dataset. Training the full deep encoder risks overfitting and instability.  
So you freeze the encoder and train a compact classifier on top.

### 7.2 Candidate models
- Calibrated Linear SVM
- RBF SVM
- MLP (small neural network)

Hyperparameters are selected with 5-fold CV on training split.

### 7.3 Selection objective
Use **macro-F1** because class-balanced performance is more meaningful than plain accuracy in small datasets.

### 7.4 Validation-first model choice
Best family is chosen by **validation macro-F1**, then reported on held-out test.

---

## 8) Open-set rejection science (Unknown class)

Implemented in `src/recognize.py` and analyzed in `src/evaluate.py`.

Classifier gives probabilities $p_k(x)$. Final decision:

$$
\hat{y}=
\begin{cases}
\arg\max_k p_k(x), & \max_k p_k(x)\ge\tau \\
\text{Unknown}, & \text{otherwise}
\end{cases}
$$

So Unknown is not a trained class in the same way; it is a **confidence gate**.  
`src/evaluate.py` sweeps $\tau$ and compares:
- genuine acceptance rate (TAR),
- impostor rejection rate (IRR),
to propose an operating threshold.

---

## 9) Real-time inference flow

Implemented in `src/recognize.py`.

For each webcam frame:
1. detect faces,
2. estimate landmarks,
3. align face crop,
4. encode embedding,
5. predict probabilities,
6. apply $\tau$,
7. draw box+label+confidence,
8. log attendance for known identities.

Attendance logging uses date-level dedup:
- one person -> one row per day.

---

## 10) What gets saved and why

- `data/processed/_manifest.csv`  
  Source-of-truth list of processed samples.

- `data/encodings/X.npy`, `y.npy`  
  Training features and labels.

- `data/encodings/label_map.json`  
  Class-ID mapping for inference/report consistency.

- `models/classifier.pkl`  
  Trained head used at runtime.

- `models/grid_results.json`  
  Hyperparameter-search evidence.

- `models/metrics.json`  
  Validation/test metrics for report.

- `logs/attendance.csv`  
  Final operational output.

---

## 11) Why this architecture is practical

### Strengths
- Fast enough for CPU real-time scenarios.
- Small-data friendly due to transfer learning.
- Easy to add new users (capture + retrain head).
- Strong engineering traceability for academic reporting.

### Limits
- HOG detector can miss profile/small faces.
- No anti-spoofing.
- Small class count limits statistical confidence.
- Extreme lighting/occlusion still challenging.

---

## 12) How to improve scientifically

1. Replace HOG with RetinaFace/MTCNN.
2. Add liveness detection.
3. Calibrate probabilities more robustly with larger validation sets.
4. Increase enrollment diversity (lighting/camera/background).
5. Expand classes and evaluate uncertainty calibration.

---

## 13) Quick viva summary (30-second version)

SmartPresence is a transfer-learning attendance pipeline: HOG detection + landmark alignment + pretrained face embeddings + lightweight SVM/MLP classifier. We train only the head on our local 3-person dataset, select hyperparameters by CV, and use a confidence threshold $\tau$ for Unknown rejection. This gives a practical real-time CPU system with reproducible artifacts (`metrics.json`, `grid_results.json`, `attendance.csv`) and a clear upgrade path to stronger detectors and liveness checks.
