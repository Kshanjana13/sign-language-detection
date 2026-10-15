"""
data_collection.py

Collects hand-landmark training samples from your webcam for a given
sign/word label and appends them to data/landmarks.csv.

USAGE:
    python src/data_collection.py --label hello --samples 300
    python src/data_collection.py --label thank_you --samples 300
    python src/data_collection.py --label yes --samples 300
    ... repeat for every word you want the model to recognize ...

CONTROLS (while the webcam window is focused):
    SPACE -> start/pause capturing samples for the current label
    q     -> quit the script

Collect data for each sign under a few different angles, hand positions,
and lighting conditions for a more robust model. 200-400 samples per
sign is usually enough for a demo-quality project.
"""

import argparse
import csv
import os
import time

import cv2
import mediapipe as mp

from model_utils import landmarks_to_row, feature_column_names

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")


def ensure_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(feature_column_names() + ["label"])


def main():
    parser = argparse.ArgumentParser(description="Collect hand-sign training data.")
    parser.add_argument("--label", required=True, help="Word/sign name, e.g. 'hello'")
    parser.add_argument("--samples", type=int, default=300, help="How many samples to collect")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    ensure_csv_header()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera index / permissions.")

    collecting = False
    collected = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands, open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting and collected < args.samples:
                    row = landmarks_to_row(hand_landmarks, label=args.label)
                    writer.writerow(row)
                    collected += 1
                    # tiny delay so we don't capture near-duplicate frames
                    time.sleep(0.03)

            # ---- HUD ----
            status = "COLLECTING" if collecting else "PAUSED (press SPACE)"
            color = (0, 200, 0) if collecting else (0, 0, 255)
            cv2.putText(frame, f"Label: {args.label}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Samples: {collected}/{args.samples}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE: start/pause   Q: quit", (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Data Collection - Sign Language", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                collecting = not collecting
            elif key == ord("q") or collected >= args.samples:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {collected} samples for label '{args.label}'.")
    print(f"Saved to {os.path.abspath(CSV_PATH)}")


if __name__ == "__main__":
    main()
