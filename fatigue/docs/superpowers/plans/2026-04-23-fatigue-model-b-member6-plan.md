# Fatigue Model B (Member 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a demo-ready fatigue detection algorithm B for member 6 that outputs `Normal`, `Yawning`, `Looking Around`, and `Fatigued Driving` through the team's unified interface.

**Architecture:** Use a pretrained feature source plus a lightweight self-trained temporal fusion model. Extract face landmarks and per-frame drowsiness-related features, aggregate them over short windows, train a small BiLSTM or GRU model, then map scores to the unified four-class output.

**Tech Stack:** Python, OpenCV, MediaPipe Face Landmarker, PyTorch, pandas, scikit-learn, matplotlib

---

## Recommended File Structure

**Files:**
- Create: `data/fatigue_b/raw/README.md`
- Create: `data/fatigue_b/processed/README.md`
- Create: `data/fatigue_b/labels/train.csv`
- Create: `data/fatigue_b/labels/val.csv`
- Create: `data/fatigue_b/labels/test.csv`
- Create: `src/fatigue_b/extract_features.py`
- Create: `src/fatigue_b/build_windows.py`
- Create: `src/fatigue_b/dataset.py`
- Create: `src/fatigue_b/model.py`
- Create: `src/fatigue_b/train.py`
- Create: `src/fatigue_b/eval.py`
- Create: `src/fatigue_b/infer.py`
- Create: `src/fatigue_b/label_map.yaml`
- Create: `outputs/fatigue_b/checkpoints/README.md`
- Create: `outputs/fatigue_b/reports/README.md`
- Create: `docs/member6-fatigue-b-report.md`

## Unified Labels

- `Normal`
- `Yawning`
- `Looking Around`
- `Fatigued Driving`

## Internal Training Targets

- `is_yawning`
- `is_look_away`
- `is_fatigued`

## Input Feature Set

- `ear`
- `mar`
- `head_yaw`
- `head_pitch`
- `head_roll`
- `drowsy_prob`

## Output Contract

```json
{
  "module": "fatigue",
  "model_name": "fatigue_b",
  "frame_id": 128,
  "timestamp": 5.12,
  "label": "Fatigued Driving",
  "confidence": 0.86,
  "indicators": {
    "yawn_score": 0.21,
    "look_away_score": 0.45,
    "fatigue_score": 0.86
  },
  "risk_level": "high"
}
```

## Day-by-Day Execution Checklist

### Task 1: Day 1 - Lock Scope, Data, and Labels

**Files:**
- Create: `data/fatigue_b/raw/README.md`
- Create: `data/fatigue_b/labels/train.csv`
- Create: `data/fatigue_b/labels/val.csv`
- Create: `data/fatigue_b/labels/test.csv`
- Create: `src/fatigue_b/label_map.yaml`

- [ ] Confirm with member 2 that the shared fatigue datasets are `NTHU-DDD`, `YawDD`, `UTA-RLDD`, and optional self-recorded demo clips.
- [ ] Freeze the label mapping:
  - `YawDD yawning -> is_yawning=1`
  - `NTHU looking aside -> is_look_away=1`
  - `NTHU drowsy/sleepy/nodding -> is_fatigued=1`
  - `UTA-RLDD low vigilance/drowsiness -> is_fatigued=1`
  - everything else -> `Normal`
- [ ] Split train/val/test by `subject_id`, never by random frame shuffle.
- [ ] Write `src/fatigue_b/label_map.yaml` with:

```yaml
public_labels:
  0: Normal
  1: Yawning
  2: Looking Around
  3: Fatigued Driving

internal_targets:
  - is_yawning
  - is_look_away
  - is_fatigued

risk_map:
  Normal: low
  Yawning: medium
  Looking Around: medium
  Fatigued Driving: high
```

- [ ] Export the final label CSV schema:

```csv
video_id,start_frame,end_frame,subject_id,source_dataset,is_yawning,is_look_away,is_fatigued,split
```

- [ ] Deliverables by end of Day 1:
  - final label standard
  - final split files
  - one-page note stating exactly which videos belong to train/val/test

### Task 2: Day 2 - Build Feature Extraction Pipeline

**Files:**
- Create: `src/fatigue_b/extract_features.py`
- Create: `data/fatigue_b/processed/README.md`

- [ ] Read each video frame-by-frame with OpenCV.
- [ ] Run MediaPipe Face Landmarker on each frame.
- [ ] Compute per-frame features:
  - `ear`
  - `mar`
  - `head_yaw`
  - `head_pitch`
  - `head_roll`
- [ ] If using a pretrained drowsiness checkpoint, also save `drowsy_prob`.
- [ ] Store one CSV per video with this schema:

```csv
frame_id,timestamp,ear,mar,head_yaw,head_pitch,head_roll,drowsy_prob,face_valid
```

- [ ] Add failure handling:
  - if no face is detected, mark `face_valid=0`
  - do not crash on a bad frame
- [ ] Validate on 3 videos:
  - one normal
  - one yawning
  - one fatigued
- [ ] Deliverables by end of Day 2:
  - feature CSVs for sample videos
  - screenshot or table showing features change with yawning and looking away

### Task 3: Day 3 - Build Window Dataset and Train the Model

**Files:**
- Create: `src/fatigue_b/build_windows.py`
- Create: `src/fatigue_b/dataset.py`
- Create: `src/fatigue_b/model.py`
- Create: `src/fatigue_b/train.py`
- Create: `outputs/fatigue_b/checkpoints/README.md`

- [ ] Convert per-frame features into fixed windows:
  - window length: `16` frames or `2 seconds`
  - stride: `8` frames
- [ ] Build one sample as:

```text
[window_size, feature_dim] = [16, 6]
```

- [ ] Create labels for each window:
  - if `is_fatigued=1`, window target includes fatigue
  - else if `is_yawning=1`, window target includes yawning
  - else if `is_look_away=1`, window target includes look away
- [ ] Implement `BiLSTM` or `GRU` with 3 sigmoid heads:
  - `yawn_head`
  - `look_away_head`
  - `fatigue_head`
- [ ] Train with:
  - optimizer: `Adam`
  - learning rate: `1e-4`
  - batch size: `16`
  - epoch count: `20-30`
  - loss: sum of 3 binary cross-entropy losses
- [ ] Save the best checkpoint by validation F1:

```text
outputs/fatigue_b/checkpoints/best_model.pt
```

- [ ] Deliverables by end of Day 3:
  - first runnable training script
  - first best checkpoint
  - training loss curve

### Task 4: Day 4 - Evaluate, Tune, and Freeze Thresholds

**Files:**
- Create: `src/fatigue_b/eval.py`
- Create: `outputs/fatigue_b/reports/README.md`

- [ ] Run evaluation on the held-out test set.
- [ ] Report:
  - precision
  - recall
  - F1-score
  - confusion matrix
- [ ] Tune thresholds for:
  - `yawn_score`
  - `look_away_score`
  - `fatigue_score`
- [ ] Freeze the final mapping logic:

```text
if fatigue_score > T_fatigue:
    label = Fatigued Driving
elif yawn_score > T_yawn:
    label = Yawning
elif look_away_score > T_look:
    label = Looking Around
else:
    label = Normal
```

- [ ] Run 3 short demo videos and record whether the output matches expectation.
- [ ] Deliverables by end of Day 4:
  - final threshold table
  - final metrics table
  - 3 demo-video predictions

### Task 5: Day 5 - Integrate with Member 7 and Prepare Defense

**Files:**
- Create: `src/fatigue_b/infer.py`
- Create: `docs/member6-fatigue-b-report.md`

- [ ] Implement one public inference function:

```python
def predict_window(feature_window):
    return {
        "module": "fatigue",
        "model_name": "fatigue_b",
        "label": "Fatigued Driving",
        "confidence": 0.86,
        "indicators": {
            "yawn_score": 0.21,
            "look_away_score": 0.45,
            "fatigue_score": 0.86
        },
        "risk_level": "high"
    }
```

- [ ] Hand off to member 7:
  - checkpoint path
  - inference script
  - sample JSON output
  - threshold file
- [ ] Write one defense page with these sections:
  - objective
  - input
  - model structure
  - training data
  - output interface
  - strengths and weaknesses
- [ ] Prepare 3 answer-ready statements:
  - why use temporal modeling
  - how this differs from member 5
  - why outputs are unified with member 7
- [ ] Deliverables by end of Day 5:
  - final inference module
  - final demo output
  - final PPT content for member 6

## Defense Talking Points

- Member 6 is not building a new paper-level architecture from scratch.
- Member 6 is building a learning-based fatigue module using pretrained perception plus self-trained temporal fusion.
- The model uses short temporal windows because fatigue is a continuous process, not a single-frame event.
- The external interface is identical to member 5 to simplify system integration and A/B comparison.

## Minimum Acceptable Result

- Can run on local demo videos without crashing.
- Can output the unified four labels.
- Can show at least one quantitative evaluation table.
- Can provide one stable inference function to member 7.

## Stretch Goal

- Export ONNX for faster deployment.
- Add score smoothing to reduce output flicker.
- Add a short ablation table comparing with and without `drowsy_prob`.

