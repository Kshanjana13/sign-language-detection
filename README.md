# Sign Language Detection (Webcam, Word + Confidence + Audio)

A real-time sign language recognition project. Show a hand sign to your
webcam and the app will:

- Detect and track your hand (MediaPipe)
- Predict the corresponding **word**
- Display a live **confidence level** (%)
- **Speak the word out loud** (offline text-to-speech)
- Build up a running sentence of confirmed words on screen

This is a landmark-based approach (not raw-pixel deep learning), which
means it trains in seconds on a normal laptop CPU and works well for a
college/course ML project.

---

## 1. Project structure

```
sign_language_detection/
├── requirements.txt
├── README.md
├── data/
│   └── landmarks.csv        <- created by data_collection.py
├── models/
│   ├── sign_model.pkl       <- created by train_model.py
│   └── label_encoder.pkl    <- created by train_model.py
└── src/
    ├── model_utils.py       <- shared landmark feature extraction
    ├── data_collection.py   <- STEP 1: record training samples
    ├── train_model.py       <- STEP 2: train the classifier
    └── real_time_detection.py  <- STEP 3: run live detection
```

## 2. Setup

Requires Python 3.9–3.11 (MediaPipe does not yet support the very
latest Python versions on all platforms).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Step 1 — Collect training data

Run this once **per sign/word** you want the model to recognize.
Aim for 200–400 samples per sign, varying hand angle/position slightly.

```bash
cd src
python data_collection.py --label hello --samples 300
python data_collection.py --label thank_you --samples 300
python data_collection.py --label yes --samples 300
python data_collection.py --label no --samples 300
python data_collection.py --label please --samples 300
python data_collection.py --label i_love_you --samples 300
```

- Press **SPACE** to start/pause capturing.
- Press **Q** to stop early.
- All samples are appended to `data/landmarks.csv`.
- Use underscores instead of spaces in multi-word labels (e.g. `thank_you`) —
  the app automatically converts them back to spaces for speech/display.

> Tip: You can re-run this command for the same label later to add more
> samples (e.g. under different lighting) — it appends, not overwrites.

## 4. Step 2 — Train the model

```bash
python train_model.py
```

This trains a Random Forest classifier on your collected landmarks,
prints a test accuracy report, and saves:
- `models/sign_model.pkl`
- `models/label_encoder.pkl`

Re-run this any time you add more training data.

## 5. Step 3 — Run real-time detection

```bash
python real_time_detection.py
```

- Show a trained sign to the webcam.
- The predicted **word** and **confidence %** appear at the top, with a
  color-coded confidence bar (green = confident, orange = below threshold).
- Once a prediction is stable for enough frames, it's spoken aloud and
  added to the sentence bar at the bottom.

**Controls:**
| Key | Action |
|-----|--------|
| C   | Clear the current sentence |
| M   | Mute / unmute audio |
| Q   | Quit |

## 6. Tuning

All the key thresholds live at the top of `real_time_detection.py`:

- `CONFIDENCE_THRESHOLD` (default 0.75) — minimum model confidence to
  accept a prediction at all.
- `STABILITY_WINDOW` / `STABILITY_MIN_AGREE` — how many consecutive
  frames must agree before a word is "confirmed" and spoken. Raise these
  if you're getting false triggers; lower them for faster response.
- `SPEAK_COOLDOWN_SECONDS` — minimum gap between spoken words.

## 7. Extending the project

- **More signs:** just collect more labels and retrain — no code changes needed.
- **Two-handed signs:** `model_utils.py` currently extracts features for
  one hand; you can extend `extract_landmark_features` to concatenate
  landmarks from `results.multi_hand_landmarks[0]` and `[1]` for 126
  features total, and set `max_num_hands=2`.
- **Deep learning version:** swap `RandomForestClassifier` in
  `train_model.py` for a small Keras MLP (`Dense` layers with softmax
  output) if your coursework requires a neural network specifically —
  the landmark feature pipeline stays the same.
- **GUI:** the OpenCV window can be swapped for a Tkinter/PyQt front end
  if you want buttons instead of keyboard shortcuts.

## 8. Troubleshooting

- **Webcam doesn't open:** try `--camera 1` (or another index) if you
  have multiple cameras.
- **MediaPipe install issues:** make sure you're on Python 3.9–3.11 and
  using a fresh virtual environment.
- **No sound:** on Linux, `pyttsx3` needs `espeak` installed
  (`sudo apt install espeak`). On Windows/macOS it uses the built-in
  system TTS voices out of the box.
- **Low accuracy:** collect more samples per sign, vary hand angle and
  distance from camera while collecting, and make sure lighting is decent.
