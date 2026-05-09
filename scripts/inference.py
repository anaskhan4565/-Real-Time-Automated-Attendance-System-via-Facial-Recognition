"""Inference script for SmartPresence: run model on single frames.

Usage:
    from scripts.inference import run_inference_on_frame
    run_inference_on_frame(image)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from src import config
from src.encode import encode_with_facenet, load_facenet_model
from src.face import build_mtcnn, detect_faces
from src.utils import align_face, draw_label


def run_inference_on_frame(image: np.ndarray, output_path: str | None = None) -> str:
    """Run inference on a single frame and save outlined result to inference_results/.

    Args:
        image: Input image as BGR numpy array (OpenCV format).
        output_path: Optional custom output path. If None, uses timestamp.

    Returns:
        Path to the saved result image.
    """
    # Ensure inference_results directory exists
    results_dir = config.PROJECT_ROOT / "inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model and label map
    model_payload = joblib.load(config.PATHS["models"] / "classifier.pkl")
    classifier = model_payload["model"]
    label_map_path = model_payload.get("label_map_path", "data/encodings/label_map.json")
    with (config.PROJECT_ROOT / label_map_path).open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Load FaceNet model
    facenet_model, facenet_device, torch_module = load_facenet_model()

    # Convert BGR to RGB for processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    mtcnn = build_mtcnn()
    boxes, landmarks_list = detect_faces(mtcnn, rgb_image)
    print(f"Detected {len(boxes)} face(s) in image")

    # Process each detected face
    for box, landmarks in zip(boxes, landmarks_list):
        try:
            # Align face
            aligned = align_face(rgb_image, box, landmarks, config.IMG_SIZE)

            # Encode embedding
            embedding = encode_with_facenet(aligned, facenet_model, facenet_device, torch_module)
            embedding = np.expand_dims(embedding, axis=0)  # Add batch dimension

            # Classify
            proba = classifier.predict_proba(embedding)[0]
            pred_id = int(np.argmax(proba))
            confidence = proba[pred_id]

            if confidence >= config.TAU:
                name = label_map["id_to_display"].get(str(pred_id), str(pred_id))
                text = f"{name}"
                color = (0, 255, 0)  # Green for known
            else:
                text = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # Draw on image
            draw_label(image, box, text, color, confidence)

        except Exception as e:
            # Skip failed faces
            print(f"Failed to process face: {e}")
            continue

    # Save result
    if output_path is None:
        timestamp = int(time.time() * 1000)  # Milliseconds
        result_path = results_dir / f"inference_{timestamp}.png"
    else:
        result_path = Path(output_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(result_path), image)

    return str(result_path)


def main() -> None:
    """Process all images in test_images folder and save results."""
    test_images_dir = config.PROJECT_ROOT / "test_images"
    if not test_images_dir.exists():
        print(f"test_images directory not found: {test_images_dir}")
        return

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_files = [
        f for f in test_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"No image files found in {test_images_dir}")
        return

    print(f"Processing {len(image_files)} images from {test_images_dir}")

    for image_file in image_files:
        print(f"Processing: {image_file.name}")

        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue

        # Generate output path
        stem = image_file.stem
        output_path = config.PROJECT_ROOT / "inference_results" / f"{stem}__result.png"

        # Run inference
        try:
            result_path = run_inference_on_frame(image, str(output_path))
            print(f"Saved result: {result_path}")
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

    print("Batch inference completed.")


if __name__ == "__main__":
    main()