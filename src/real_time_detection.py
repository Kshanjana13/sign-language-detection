"""
real_time_detection.py

Real-time sign language detection app.

- Opens the webcam
- Tracks your hand with MediaPipe
- Predicts the word/sign using the trained model
- Displays the predicted word + confidence level on screen
- Speaks the recognized word out loud (offline TTS, no internet needed)
- Builds up a running "sentence" of confirmed words

USAGE:
    python src/real_time_detection.py

CONTROLS:
    c -> clear the sentence
    m -> mute/unmute audio
    q -> quit
"""

import os
import queue
import threading
import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

from model_utils import extract_landmark_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

CONFIDENCE_THRESHOLD = 0.75      # minimum confidence to consider a prediction valid
STABILITY_WINDOW = 10            # how many recent frames to look at for a stable vote
STABILITY_MIN_AGREE = 7          # how many of those frames must agree to "confirm" a word
SPEAK_COOLDOWN_SECONDS = 1.5     # minimum time between spoken words


class SpeechWorker:
    """Runs pyttsx3 on a background thread so speaking never blocks the video loop."""

    def __init__(self):
        self.q = queue.Queue()
        self.muted = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while True:
            word = self.q.get()
            if word is None:
                break
            if not self.muted:
                engine.say(word.replace("_", " "))
                engine.runAndWait()

    def speak(self, word):
        self.q.put(word)

    def toggle_mute(self):
        self.muted = not self.muted


def draw_confidence_bar(frame, x, y, w, h, confidence):
    """Draws a simple filled confidence bar (0.0 - 1.0)."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 2)
    fill_w = int(w * max(0.0, min(1.0, confidence)))
    color = (0, 200, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run data_collection.py then train_model.py first."
        )

    clf = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    speech = SpeechWorker()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    recent_predictions = deque(maxlen=STABILITY_WINDOW)
    last_spoken_word = None
    last_spoken_time = 0.0
    sentence = []

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display_word = "..."
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                features = extract_landmark_features(hand_landmarks).reshape(1, -1)
                probs = clf.predict_proba(features)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                predicted_label = encoder.inverse_transform([best_idx])[0]

                if confidence >= CONFIDENCE_THRESHOLD:
                    display_word = predicted_label
                    recent_predictions.append(predicted_label)
                else:
                    display_word = "..."
                    recent_predictions.append(None)
            else:
                recent_predictions.append(None)

            # ---- Stability check: only "confirm" and speak a word once it has
            # dominated the recent frame window, to avoid flicker / false triggers ----
            valid_recent = [p for p in recent_predictions if p is not None]
            if valid_recent:
                most_common_word, count = Counter(valid_recent).most_common(1)[0]
                now = time.time()
                if (
                    count >= STABILITY_MIN_AGREE
                    and most_common_word != last_spoken_word
                    and (now - last_spoken_time) >= SPEAK_COOLDOWN_SECONDS
                ):
                    speech.speak(most_common_word)
                    sentence.append(most_common_word)
                    last_spoken_word = most_common_word
                    last_spoken_time = now
                    recent_predictions.clear()

            # ================= HUD / overlay =================
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            cv2.putText(frame, f"Word: {display_word}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            draw_confidence_bar(frame, 300, 55, 250, 20, confidence)

            mute_label = "MUTED" if speech.muted else "AUDIO ON"
            cv2.putText(frame, mute_label, (w - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Sentence bar at bottom
            sentence_text = " ".join(sentence[-10:]).replace("_", " ")
            cv2.rectangle(frame, (0, h - 45), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, sentence_text if sentence_text else "(sentence appears here)",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.putText(frame, "C: clear   M: mute   Q: quit", (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Sign Language Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sentence.clear()
            elif key == ord("m"):
                speech.toggle_mute()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
