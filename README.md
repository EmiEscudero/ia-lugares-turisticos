# Tourist place recognition in Mexico

[Español](README.es.md) · English

Recognize a **destination and its state** from a photograph. The new `tourist_places` workflow uses a pretrained EfficientNet-B0, canonical destination labels, duplicate-aware train/validation/test splits, and CUDA mixed precision. The original notebooks and GUI are retained as historical experiments; their checkpoints are not compatible with this new model.

Read [the project review](docs/REVIEW.md) before interpreting the original accuracy numbers. They describe beach-versus-town accuracy, not correct-place accuracy.

The [real-data pilot](docs/EXPERIMENTS.md) covers **164 destinations** after deduplication and conflict quarantine. The selected model achieves **31.6% test top-1 / 54.5% top-5**, versus **1.9% / 11.5%** for an adapted original-CNN baseline on the same 209 test photos. This is an exploratory comparison, not coverage of all 480 source destinations. Training photos remain private, and trained weights are excluded from Git; cloning the code does not download a trained model. The prepared cloud deployment obtains sanitized model assets from a GitHub release.

## Setup on Windows / RTX 4060

Run PowerShell in the project folder. Python 3.13 was used for validation. Dependencies stay in `.venv`; downloaded weights stay in `artifacts/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The last command should print `True` and the NVIDIA GPU name. For a different platform, select an appropriate build using the [official PyTorch installer](https://pytorch.org/get-started/locally/). The two pinned versions above are the CUDA pair installed for this project; `requirements.txt` expresses the API compatibility range.

## Prepare images

Supply an authorized local copy of the training image collection. Private download links are not published in the current documentation. Removing those links does not change Drive access permissions; earlier Git commits may retain previous links. Place both category folders inside one source folder, for example:

```text
data/raw/
  PLAYAS/
    QUINTANA_ROO/
      TULUM/
        train/*.jpg
        validation/*.jpg
  PUEBLOS/
    JALISCO/
      TEQUILA/
        train/*.jpg
        validation/*.jpg
```

The source may be elsewhere on disk; provide that folder with `--source`. Images are read in place, never modified or copied by preparation. Old `train` and `validation` folders are pooled and split again, because the original split contains duplicate photos across its boundary. The historical flat CSVs are **not** used to assign new labels.

```powershell
.\.venv\Scripts\python.exe -m tourist_places.audit --output artifacts/legacy_audit.json
.\.venv\Scripts\python.exe -m tourist_places.prepare --source data/raw --output data/prepared --min-images 6
```

Review `data/prepared/preparation_report.json` before training:

- Labels come from normalized `(state, destination)` folder names. A place found under both categories becomes one destination with both type tags. Names and geographic boundaries still need human review: a region such as Riviera Maya overlaps more specific destinations.
- Matching image families cannot carry conflicting destination labels; preparation stops and records conflicts for correction. For an exploratory run, `--exclude-conflicts` explicitly quarantines every image in those families, without guessing the correct label. No source image is deleted or changed.
- Exact duplicates and similar re-encoded or mirrored photos are grouped across the entire collection before splitting. One representative per nonconflicting family is retained. Near matching uses a 64-bit difference hash with a default distance of 4; inspect groups for accidental matches, especially generic horizons. Use `--hash-distance -1` to disable near matching while retaining exact decoded-content checks. Cropped views and source independence still need human review.
- Destinations with fewer than 6 independent groups are excluded and listed. This is a pilot filter, not a claim that 6 photos are sufficient for reliable recognition. Collect more independent photos before expanding coverage. Mirrored copies do not count as new evidence.
- Included destinations receive approximately 70% training, 15% validation, and 15% test images, with at least one in each split. A saved seed and manifest digest identify the split. An existing manifest cannot be overwritten.
- For stronger evaluation, manually group photos from the same photographer, visit, or source before making a final benchmark. Visual duplicate matching cannot establish source independence.

If preparation excludes most destinations, start with a smaller, well-photographed set and collect more photos. As an initial collection goal, aim for 50–100 independent photos per destination with varied viewpoints, weather, lighting, and cameras. The required count depends on visual distinctiveness; a generic sand-and-water photo may not contain enough information to name a specific beach.

## Train and compare

```powershell
.\.venv\Scripts\python.exe -m tourist_places.train --device cuda --prepared data/prepared --output runs/first --batch-size 16 --epochs 25
```

The default trains the new classifier for 3 epochs with the pretrained backbone frozen, then fine-tunes the backbone at a lower learning rate. It uses 224-pixel inputs, training-only augmentation, ImageNet normalization, AdamW, a learning-rate schedule, and early stopping. The best checkpoint is selected by validation macro-F1, with validation loss breaking ties. Pretrained BatchNorm running statistics stay fixed during fine-tuning; `--update-batchnorm` enables an alternative to compare on validation data.

`history.csv` records **destination** top-1/top-5 accuracy, macro-F1, macro-recall, loss, seconds per epoch, and peak allocated VRAM. Accuracy is aggregated by image count rather than averaging batch percentages. A run records its settings, class mapping, and split identity. Seeds make the data split repeatable; CUDA training is not guaranteed bit-for-bit deterministic.

Start with batch size 16 for the 8 GB RTX 4060. Try 32 after measuring VRAM; use 8 if necessary. Data loading defaults to two worker processes and pinned memory on CUDA. Use `--workers 0` to diagnose Windows worker issues. `--no-amp` allows a full-precision comparison. `--no-pretrained` exists for offline smoke tests and should not be used for the real baseline.

Measure a few warmed-up epochs to establish speed. Worker startup adds overhead to the first epoch. Convolution autotuning is off by default for quick startup on small datasets; `--cudnn-benchmark` opts into it and may add substantial setup time to the first epoch and the first unfrozen epoch. Neither the GPU name nor a synthetic smoke test establishes a speedup factor or destination accuracy on this dataset.

For a comparison with the original network design on exactly the same destination labels and split:

```powershell
.\.venv\Scripts\python.exe -m tourist_places.train --architecture legacy_cnn --device cuda --prepared data/prepared --output runs/legacy_baseline
```

This baseline retains the original three convolutional stages and 512-unit dense layer, at 128 pixels and without ImageNet normalization. It replaces the hierarchical outputs with one destination head, uses the same cleaned split and training-only augmentation, and trains all weights from scratch. It is a controlled legacy-style baseline, not the original notebook's beach/town accuracy.

## Evaluate and predict

Use validation for decisions about settings. Once the model and dataset are fixed, evaluate the held-out test split; do not tune settings repeatedly against test results.

```powershell
.\.venv\Scripts\python.exe -m tourist_places.evaluate --checkpoint runs/first/best.pt --prepared data/prepared --split val
.\.venv\Scripts\python.exe -m tourist_places.evaluate --checkpoint runs/first/best.pt --prepared data/prepared --split test
.\.venv\Scripts\python.exe -m tourist_places.predict "C:\photos\example.jpg" --checkpoint runs/first/best.pt --top-k 5
```

Evaluation saves aggregate metrics, per-destination recall, and nonzero confusion counts. `--data-root` on training/evaluation relocates the source images without changing the saved split. Predictions return the saved name, state, and type tags. Scores below `--min-confidence` (default 0.5) are marked uncertain. This threshold is a display heuristic: softmax scores are not calibrated certainty, and high confidence does not prove that an unknown place belongs to a known class. Validate rejection behavior using unseen places before deploying it.

The original `predictGUI.py` still loads the historical `modeloCNN.pth`. Use the new CLI or Streamlit interface for the new checkpoints.

## Local Streamlit interface

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open [the local app](http://127.0.0.1:8501). It defaults to **Spanish**; use **Idioma / Language** at the top to switch to English. Direct links can select [Spanish](http://127.0.0.1:8501/?lang=es) or [English](http://127.0.0.1:8501/?lang=en). Changing language keeps the selected photo, model, threshold, and prediction.

Upload a JPEG, PNG, or WebP photo, then choose **Encontrar este lugar / Find this place**. The app shows ranked destinations and states, percentage scores, uncertain matches, downloadable results, the model's measured quality, and its destination coverage. Trained models under `runs/<name>/best.pt` are discovered when refreshed. Smoke models are excluded. Optional examples under `data/ui_examples/` should come from training or validation, never from the held-out test set.

The four pilot variants appear in a fixed order, by **validation macro-F1, highest first**:

| Order | Model | Purpose | Validation macro-F1 |
| --- | --- | --- | ---: |
| 1 | Stable fine-tuning · `stable_transfer_pilot` | Recommended and selected by default; adapts EfficientNet with fixed BatchNorm statistics. | 0.310 |
| 2 | Frozen features · `frozen_pilot` | Trains the classifier while keeping the pretrained feature network fixed. | 0.287 |
| 3 | Initial fine-tuning · `transfer_pilot` | Initial comparison with updated BatchNorm statistics and convolution autotuning. | 0.228 |
| 4 | Adapted original CNN · `legacy_pilot` | Baseline trained from scratch using the original CNN design. | 0.029 |

These numbers identify the **option order**, not accuracy or a prediction score. `runs/recommended/best.pt` is a local copy of the selected stable model, not a fifth model. A verified matching copy is used for the first option. Additional runs appear after known variants, sorted by name. The model picker is visible on the main page; **About the model** explains each variant and compares its measurements. Selection used validation before the test evaluation; the recommended model is not changed to favor the highest observed test score.

**About the model** also shows a description and two diagrams that change with the selected model: how its training and evaluation were organized, and how an uploaded photo becomes ranked predictions. The diagrams distinguish EfficientNet's pretrained features and 224-pixel input from the original CNN's 128-pixel input and training from scratch. Pilot data counts are shown only for a checkpoint with the recorded pilot split identity. These are schematic process explanations, not visual explanations of which pixels influenced a particular prediction.

The local command above binds the server only to this computer. Uploads stay in memory and are not saved by the application. A small local component provides accessible Spanish/English upload controls, without external scripts or persistent browser storage. The server also validates file size, decoded format, and pixel count. Inference runs on CPU to leave the GPU free for training. Model caching includes the checkpoint modification time, so replacing a checkpoint does not reuse stale weights. The original Drive folder is a private data source; the interface has no Drive editing capability.

The cloud deployment is prepared to use `cloud/streamlit_app.py` from the development branch `dev/improve-place-recognition`, with Python 3.13 and separate CPU-only dependencies. It downloads four sanitized model assets from a GitHub release and verifies each against its pinned SHA-256 digest. Weights remain outside Git, and no source photos are uploaded for deployment. See [deployment instructions](docs/DEPLOYMENT.md) for the configuration and release assets.

**The `main` branch is not required**: Streamlit Community Cloud lets you select a repository, branch, and entrypoint, as its [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) explains. The local command above continues to serve only on localhost; running it does not publish the app or send changes to GitHub.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/uploader.test.cjs
```

The Python checks cover data integrity, metrics, saved model compatibility, upload validation, missing/broken model states, stable model ordering and verified aliases, model explanations, bilingual copy, and preserved predictions across language changes. Four additional browser-component checks (Node.js, optional for using the app) cover upload events, language changes, file limits, clearing, and overlapping reads. Images, environments, downloaded weights, and trained checkpoints are excluded from Git. Aggregate pilot measurements and the destination mapping are published under `experiments/pilot/`. Detailed image manifests, source inventories, and per-image preparation reports are kept locally because they can include private source identifiers.
