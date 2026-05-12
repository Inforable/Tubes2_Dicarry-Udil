# Development Workflow — IF3270 Tugas Besar 2

## Team Members
- Member A (NIM: 18223121)
- Member B (NIM: 13523160)
- Member C (NIM: 13523156)

## Work Distribution

### Member A — CNN Track Lead

Responsible for the entirety of Track A (CNN Image Classification).

**Primary deliverables:**
1. `src/shared/activations.py` — ReLU, Softmax, Sigmoid, Tanh implementations
2. `src/cnn/utils.py` — all three utility functions (load_image, load_batch, extract_and_save_features)
3. `src/cnn/layers.py` — all from-scratch CNN layer classes (Conv2D, LocallyConnected2D, all Pooling variants, Flatten)
4. `src/cnn/model.py` — CNNFromScratch class
5. `src/cnn/train.ipynb` — training all 16 Keras model variants, saving weights and histories
6. `src/cnn/evaluate.ipynb` — all Track A experiments and analysis
7. Track A sections of the final report (`doc/`)

**Dependency note:** Member A must complete `src/shared/activations.py` and `src/cnn/utils.py` before Member C begins feature extraction for Track B.

---

### Member B — RNN Decoder Lead

Responsible for the RNN decoder pipeline in Track B.

**Primary deliverables:**
1. `src/captioning/common/preprocessing.py` — full caption preprocessing pipeline (tokenization, vocab building, padding)
2. `src/captioning/rnn/layers.py` — EmbeddingLayer and SimpleRNNCell from-scratch implementations
3. `src/captioning/rnn/model.py` — RNNCaptioner class (full pipeline from image path to caption string)
4. `src/captioning/rnn/train.ipynb` — 6+ RNN training runs, weight saving
5. `src/captioning/rnn/evaluate.ipynb` — BLEU-4, METEOR, Keras vs scratch comparison for RNN
6. RNN sections of the final report

**Dependency note:** Member B requires `data/flickr8k/features.npy` (produced by Member C) before training can begin. Preprocessing and layer implementation can be done in parallel.

---

### Member C — LSTM Decoder Lead + Feature Extraction

Responsible for the LSTM decoder pipeline and the shared CNN encoder for Track B.

**Primary deliverables:**
1. `src/captioning/common/feature_extraction.py` — pretrained CNN encoder setup and feature extraction for all Flickr8k images, saves `features.npy` and `features_index.json`
2. `src/shared/dense.py` — Dense layer from-scratch (ported from Tubes 1)
3. `src/captioning/lstm/layers.py" — LSTMCell, DenseProjection, DenseOutput from-scratch implementations
4. `src/captioning/lstm/model.py` — LSTMCaptioner class
5. `src/captioning/lstm/train.ipynb` — 6+ LSTM training runs, weight saving
6. `src/captioning/lstm/evaluate.ipynb` — BLEU-4, METEOR, Keras vs scratch comparison for LSTM, qualitative analysis (10 example images with RNN vs LSTM vs ground truth captions), max caption length experiments
7. LSTM sections of the final report

**Dependency note:** Member C must complete feature extraction as the absolute first priority — both Member B and Member C's training pipelines depend on `features.npy`.

---

## Shared Responsibilities (all members)

- `README.md` — written collaboratively, covers setup and run instructions
- `doc/` final PDF — each member writes their own sections; Member A compiles and submits
- Cross-review: each member verifies that the other's from-scratch output matches Keras before the evaluate notebooks are finalized

## Execution order

```
Day 1-2:
  C → common/feature_extraction.py + run extraction (features.npy)   ← BLOCKS B and C training
  A → utils.py, activations.py
  B → common/preprocessing.py (can run in parallel with C)

Day 3-4:
  A → layers.py (CNN from scratch) + model.py
  B → rnn/layers.py + rnn/model.py
  C → lstm/layers.py + lstm/model.py + dense.py

Day 5:
  A → train.ipynb (16 CNN models)
  B → rnn/train.ipynb (6+ RNN models)
  C → lstm/train.ipynb (6+ LSTM models)

Day 6:
  A → evaluate.ipynb (CNN experiments + analysis)
  B → rnn/evaluate.ipynb
  C → lstm/evaluate.ipynb (including RNN vs LSTM qualitative comparison)

Day 7:
  All → report writing, final review, submission
```

## Git Conventions

- One branch per member: `feature/member-a`, `feature/member-b`, `feature/member-c`
- Merge to `main` only after self-testing
- Never commit dataset files, `features.npy`, or weight `.h5`/`.npy` files (covered by `.gitignore`)
- Share trained weights and `features.npy` via Google Drive or a shared folder — post the link in the group chat
