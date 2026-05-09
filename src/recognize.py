"""Live webcam attendance recognition for SmartPresence."""

from __future__ import annotations

import argparse
import json

import cv2
import joblib
import numpy as np

from src import config
from src.encode import encode_with_facenet, l2_normalize, load_facenet_model
from src.face import build_mtcnn, detect_faces
from src.utils import CSVAttendanceLogger, FPSMeter, align_face, draw_label, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live SmartPresence attendance recognition.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override unknown rejection threshold (default: config.TAU).",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", dest="display", action="store_true", help="Show UI.")
    display_group.add_argument("--no-display", dest="display", action="store_false", help="Disable UI.")
    parser.set_defaults(display=True)
    return parser.parse_args()


def load_model_and_labels() -> tuple[object, str, dict]:
    model_payload = joblib.load(config.PATHS["models"] / "classifier.pkl")
    classifier = model_payload["model"]
    backend = model_payload.get("backend", config.ENCODER_BACKEND)
    if backend == "face_recognition":
        backend = "facenet_pytorch"

    label_map_path = model_payload.get("label_map_path", "data/encodings/label_map.json")
    with (config.PROJECT_ROOT / label_map_path).open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return classifier, backend, label_map


def draw_overlay(frame: np.ndarray, fps: float) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    h = frame.shape[0]
    cv2.putText(
        frame,
        "Press q to quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    config.ensure_dirs()
    seed_everything(config.SEED)

    tau = float(config.TAU if args.tau is None else args.tau)
    classifier, backend, label_map = load_model_and_labels()
    logger = CSVAttendanceLogger(config.PATHS["attendance_csv"])

    facenet_model = facenet_device = facenet_torch = None
    if backend == "facenet_pytorch":
        facenet_model, facenet_device, facenet_torch = load_facenet_model()

    mtcnn = build_mtcnn()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    fps_meter = FPSMeter(alpha=0.9)
    newly_logged = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, landmarks_list = detect_faces(mtcnn, rgb)

            for box, landmarks in zip(boxes, landmarks_list):
                try:
                    aligned = align_face(
                        rgb_image=rgb,
                        face_box=box,
                        landmarks=landmarks,
                        out_size=config.IMG_SIZE,
                    )
                except Exception:
                    continue

                emb = encode_with_facenet(
                    rgb=aligned.astype(np.uint8),
                    model=facenet_model,
                    device=facenet_device,
                    torch_module=facenet_torch,
                )

                emb = l2_normalize(emb.astype(np.float32)).reshape(1, -1)
                probs = classifier.predict_proba(emb)[0]
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])

                if conf < tau:
                    name = "Unknown"
                    color = (0, 0, 255)
                else:
                    name = label_map["id_to_display"].get(str(pred_id), str(pred_id))
                    roll_no = label_map["id_to_roll"].get(str(pred_id), "unknown")
                    color = (0, 255, 0)
                    if logger.log(name=name, roll_no=roll_no, confidence=conf):
                        newly_logged += 1

                draw_label(frame, box, f"{name}", color=color, confidence=conf)

            fps_meter.update()
            draw_overlay(frame, fps_meter.fps)

            if args.display:
                cv2.imshow("SmartPresence Live Recognition", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Session over. New rows logged: {newly_logged}. CSV: {config.PATHS['attendance_csv']}")


if __name__ == "__main__":
    main()
