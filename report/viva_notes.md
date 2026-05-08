# SmartPresence Viva Notes (One-Page)

## 1) Task Definition
Our task is closed-set face identification with open-set rejection for attendance automation. In practice, the pipeline first detects faces with HOG, then classifies each aligned face into one of K=3 known identities or "Unknown" using a confidence threshold $\tau$. The decision rule is max-softmax thresholding in `src/recognize.py`, with $\tau$ selected from ROC-style analysis in `src/evaluate.py`.

## 2) Dataset
We target about 240 raw images total (roughly 80 per teammate across 3 classes), captured by webcam under nine pose/expression prompts. Preprocessing produces one aligned image plus three augmented variants per accepted raw sample (effective 4x expansion). Splits are stratified 80/10/10 as implemented in `src/train_classifier.py` using fixed `SEED=42`.

## 3) Pre-processing
Each frame is converted BGR->RGB, then HOG detection keeps only single-face samples. Landmarks are reduced to a 5-point template (eye centers, nose tip, mouth corners), and alignment is done using `cv2.estimateAffinePartial2D` + `cv2.warpAffine` into 160x160 crops. Augmentations are flip, rotation (plus/minus 15 deg), brightness scaling (0.8 to 1.2), and Gaussian noise (sigma=0.01), which improve invariance to pose, lighting, and sensor noise.

## 4) Architecture
The encoder is frozen: primary backend is dlib ResNet-34 with 128-d L2-normalized embedding; optional backend is InceptionResnetV1 with 512-d output. On top of embeddings, we train either calibrated SVM variants or an MLP head and choose the winner by validation macro-F1. Final prediction is passed through $\tau$ thresholding so uncertain faces become "Unknown" and are not logged.

## 5) Loss
Background encoder training follows triplet loss with margin $\alpha=0.2$ (one term, one weight), which explains embedding geometry but is not re-optimized in our small dataset setting. For the head, we compare cross-entropy plus L2 decay (MLP, $\lambda=10^{-4}$) against regularized hinge-loss SVM objectives with tuned $C$ (and $\gamma$ for RBF). Best-family selection is based on held-out validation macro-F1, not only CV mean.

## 6) Hyperparameters
Key values: image size 160, augmentation multiplier 3, rotation plus/minus 15, brightness 0.8-1.2, split 0.8/0.1/0.1, CV folds 5. SVM grid uses $C \in \{0.1,1,10,100\}$ and $\gamma \in \{\text{scale},0.01,0.1\}$; MLP grid uses hidden sizes {64,128} and learning rates {1e-3,1e-4} with `max_iter=200` and `early_stopping=True`. Selected $\tau$, chosen family, and concrete validation/test scores come from `models/metrics.json` and `models/grid_results.json` after running training/evaluation.

## 7) SOTA Context
Our encoder inherits published LFW context (dlib ~99.38%, FaceNet ~99.63%, ArcFace ~99.83%) but we do not claim benchmark SOTA because we do not retrain on large public identity sets. Our contribution is a deployable attendance system with small-data adaptation through a lightweight retrainable head. Concrete project scores should be quoted as `F1=<from models/metrics.json:test.f1_macro>` and `FPS=<30s average from src/recognize.py>` once artifacts are generated.

## Likely Viva Questions

**Q: Why no end-to-end training?**  
Because ~240 raw images are insufficient to reliably fine-tune a ~22M-parameter encoder without overfitting. We use transfer learning: freeze the encoder and train only a small K=3 head, which is statistically safer and computationally fast.

**Q: How is "Unknown" detected?**  
By max-softmax thresholding at $\tau$: if the highest class probability is below $\tau$, output is "Unknown". The threshold is chosen from ROC-style sweeps in `src/evaluate.py` using impostor/noisy out-of-distribution embeddings.

**Q: Why HOG over CNN detectors?**  
HOG is chosen for real-time CPU performance and simple deployment constraints. CNN detectors (MTCNN/RetinaFace) are acknowledged as future upgrades for stronger profile/small-face robustness.

**Q: How would we add a 4th student?**  
Capture ~80 frames for the new label, run preprocess -> encode -> train_classifier, then redeploy recognize. Because only the head is retrained, update time is typically under a minute on a standard laptop.
