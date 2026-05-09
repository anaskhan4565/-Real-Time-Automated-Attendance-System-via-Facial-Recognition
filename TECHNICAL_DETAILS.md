## Overall Workflow

This project is a classic face-recognition attendance pipeline with five main stages:

1. `src.capture_dataset.py`
2. `src.preprocess.py`
3. `src.encode.py`
4. `src.train_classifier.py`
5. `src.recognize.py`

Each stage builds on the previous one, from raw webcam frames to a live attendance system.

---

## 1) Registration / Data Capture

File: capture_dataset.py

### What happens
- You run:
  - `python -m src.capture_dataset --person <label>`
- `<label>` must match one of the entries in config.py under `PEOPLE`.
- The script opens your webcam and captures frames until it has enough valid images.

### Technical details
- Uses OpenCV (`cv2.VideoCapture(0)`) for webcam input and display.
- Uses `src.face.build_mtcnn()` / `detect_faces()` to verify face presence.
- Only saves a frame if exactly one large face is detected:
  - face area ≥ 10000 pixels
  - exactly one detected face
- Saved frames go to:
  - `data/raw/<person_label>/`
  - Filenames like `<label>_0001.jpg`, `<label>_0002.jpg`, etc.
- `config.PEOPLE` defines valid labels, display names, and roll numbers.

### Why this matters
- This is the “registration” step: collecting labeled face images for a person.
- It ensures the data is relatively clean before further processing.

---

## 2) Preprocessing

File: preprocess.py

### What happens
- It reads raw images from `data/raw/<person_label>/`
- Detects faces again and aligns each face
- Saves the aligned crops to `data/processed/<person_label>/`
- Optionally creates augmented versions

### Technical details
- Uses the same `MTCNN` face detector via `src.face.detect_faces()`
- Uses `src.utils.align_face()` to align using 5 facial landmarks
- Alignment target is an ArcFace-like template scaled to `IMG_SIZE = 160`
- Writes one base aligned image per raw image
- If augmentation is enabled:
  - horizontal flip
  - random rotation ±15°
  - brightness scaling
  - Gaussian noise
- Each processed file is logged in _manifest.csv

### Why this matters
- Alignment makes faces consistent for the embedding model.
- Augmentation increases data diversity and improves classifier robustness.

---

## 3) Encoding

File: encode.py

### What happens
- It reads _manifest.csv
- Loads each aligned face image
- Converts it to a 512-dimensional embedding

### Technical details
- Uses `facenet-pytorch` `InceptionResnetV1(pretrained="vggface2")`
- Input normalization is:
  - `(rgb - 127.5) / 128.0`
- Embeddings are L2-normalized to unit length
- Saves results to:
  - X.npy
  - y.npy
  - label_map.json
  - meta.json

### Why this matters
- This stage converts images into numeric feature vectors that the classifier can learn from.
- `label_map.json` preserves mapping from numeric class IDs to name/roll info.

---

## 4) Classifier Training

File: train_classifier.py

### What happens
- Loads embeddings from encodings
- Splits them into train/val/test
- Trains several classifier families
- Selects the best one and saves it

### Technical details
- Data split is controlled by `config.TRAIN_VAL_TEST = (0.8, 0.1, 0.1)`
- Classifier families:
  - calibrated `LinearSVC`
  - `SVC` with RBF kernel
  - optionally `MLPClassifier`
- Uses `StandardScaler`
- Uses `GridSearchCV` with `cv=config.CV_FOLDS`
- Selection is based on validation macro F1 score
- Saves final model payload to:
  - classifier.pkl
- Also saves:
  - grid_results.json
  - metrics.json
  - training_log.txt

### Why this matters
- The system does not do direct embedding matching at runtime.
- It trains a classifier on top of FaceNet embeddings, which is faster and easier for a small closed set of identities.

---

## 5) Live Recognition & Attendance Logging

File: recognize.py

### What happens
- Opens webcam
- Detects faces in each frame
- Aligns each face
- Encodes it with FaceNet
- Predicts identity using the trained classifier
- Logs attendance if confidence is high enough

### Technical details
- Uses `src.face.build_mtcnn()` for live face detection
- Uses `src.utils.align_face()` for alignment
- Uses `src.encode.encode_with_facenet()` + `l2_normalize()` for embeddings
- Loads classifier from classifier.pkl
- Reads label mapping from label_map.json
- Uses `predict_proba()` to get class probabilities
- Unknown rejection threshold: `config.TAU = 0.55`
- If top probability < `tau`, label is `Unknown`
- If a real person is detected:
  - draws bounding box and label
  - logs attendance via `CSVAttendanceLogger`
- Attendance is written to attendance.csv

### Attendance logging rules
- The logger de-duplicates per `(name, date)` so each person is recorded once per day.
- It writes:
  - name, roll_no, date, time, confidence

---

## Person Registration in the System

### A) Adding a new person
- Add the new person entry into config.py `PEOPLE`
  - format: `(label, display_name, roll_no)`
- Run:
  1. `python -m src.capture_dataset --person <label>`
  2. `python -m src.preprocess`
  3. `python -m src.encode`
  4. `python -m src.train_classifier`

### B) If the person label already exists
- Just capture more raw images with:
  - `python -m src.capture_dataset --person <label>`
- Then re-run preprocessing / encoding / training to update the model.

### C) What gets registered
- The system registers a person by associating:
  - `data/raw/<label>/` images
  - processed aligned crops
  - embeddings `y` labels
  - classifier class ID
  - display name + roll number from `label_map.json`

---

## Summary

- `capture_dataset` = raw labeled webcam image collection
- `preprocess` = face detection + alignment + augmentation
- `encode` = convert aligned faces into FaceNet embeddings
- `train_classifier` = learn a classifier on embeddings
- `recognize` = live camera inference + attendance logging
