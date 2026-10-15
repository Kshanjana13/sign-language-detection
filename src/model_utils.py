"""
model_utils.py
Shared helper functions used by data_collection.py, train_model.py and
real_time_detection.py.

The core idea:
  MediaPipe Hands gives us 21 (x, y, z) landmarks per detected hand.
  We flatten those into a 63-value feature vector, and normalize it
  relative to the wrist point (landmark 0) so the model is invariant
  to where the hand is positioned in the frame. This keeps the model
  small, fast, and doesn't require a GPU or a large dataset.
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURES_PER_HAND = NUM_LANDMARKS * 3  # x, y, z


def extract_landmark_features(hand_landmarks):
    """
    Convert a MediaPipe hand_landmarks object into a flat, normalized
    numpy feature vector of length 63.

    Normalization steps:
      1. Translate so the wrist (landmark 0) is the origin.
      2. Scale by the max distance from the wrist to any other landmark,
         so the feature vector is roughly invariant to hand size / distance
         from the camera.
    """
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = coords[0].copy()
    coords -= wrist  # translate relative to wrist

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 1e-6:
        coords /= max_dist  # scale invariant to hand size / distance to camera

    return coords.flatten()


def landmarks_to_row(hand_landmarks, label=None):
    """Build a CSV-ready row (list) from landmarks, optionally with a label."""
    features = extract_landmark_features(hand_landmarks)
    row = list(features)
    if label is not None:
        row.append(label)
    return row


def feature_column_names():
    cols = []
    for i in range(NUM_LANDMARKS):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    return cols
