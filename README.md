# SmartPresence: Real-Time Automated Attendance via Facial Recognition

| Name | Roll Number |
|---|---|
| Ahsan Ali | 22K-4176 |
| Mohammad Anas | 22K-4548 |
| Anas Khan | 22K-4483 |

SmartPresence is a webcam-based, CPU-friendly attendance system for the Fundamentals of Computer Vision project. It detects faces with facenet-pytorch's MTCNN, aligns faces to a canonical template, extracts pretrained FaceNet embeddings, classifies identities with an SVM/MLP head, applies open-set rejection via a confidence threshold, and logs de-duplicated attendance rows to CSV.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- encodings/
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- capture_dataset.py
|   |-- preprocess.py
|   |-- encode.py
|   |-- train_classifier.py
|   |-- evaluate.py
|   |-- recognize.py
|   `-- utils.py
|-- scripts/
|   |-- __init__.py
|   `-- make_diagrams.py
|-- models/
|-- logs/
|-- report/
|   |-- report.md
|   `-- figures/
|-- requirements.txt
`-- README.md
```

## Installation (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use Python 3.10 or 3.11 for best Windows compatibility with `torch` and `facenet-pytorch`. If install issues occur, verify your CUDA/CUDA toolkit setup or use the CPU-only wheel build for PyTorch.

## Running the Full Pipeline

1. Capture data for each teammate:

   ```powershell
   python -m src.capture_dataset --person ahsan_22K-4176
   python -m src.capture_dataset --person anas_m_22K-4548
   python -m src.capture_dataset --person anas_k_22K-4483
   ```

2. Preprocess + augment:

   ```powershell
   python -m src.preprocess
   ```

3. Encode embeddings:

   ```powershell
   python -m src.encode
   ```

4. Train classifier head:

   ```powershell
   python -m src.train_classifier
   ```

5. Evaluate + figures:

   ```powershell
   python -m src.evaluate
   python -m scripts.make_diagrams
   ```

6. Real-time attendance:

   ```powershell
   python -m src.recognize
   ```

## Inference-Only (using shipped models/classifier.pkl)

```powershell
python -m src.recognize
```

## Reproducibility

- Global seed is `SEED=42` in `src/config.py`.
- Random state is propagated to Python `random`, NumPy, and Torch through `src/utils.py`.
- Torch deterministic flags are enabled when Torch is available (`cudnn.deterministic=True`, `cudnn.benchmark=False`).
- Dependency versions are pinned in `requirements.txt` (including `facenet-pytorch==2.6.0`, `torch>=2.2,<2.5`, `scikit-learn==1.5.2`).

## Outputs

- `models/classifier.pkl` — trained head
- `models/metrics.json` — val/test metrics
- `logs/attendance.csv` — attendance log (auto-created)
- `report/report.md` — final technical report
- `report/figures/*.png` — all figures referenced in the report

## Troubleshooting

- Camera not opening: verify Windows camera privacy permissions and close apps already using the webcam.
- If install issues occur, verify your PyTorch installation and use a CPU or CUDA wheel compatible with your Python version.
- "No face detected": improve frontal lighting, reduce camera distance, and recapture samples.

## Team

- Ahsan Ali (22K-4176)
- Mohammad Anas (22K-4548)
- Anas Khan (22K-4483)

Course: Fundamentals of Computer Vision, Instructor: Dr. Kamran Ali, May 2026.
