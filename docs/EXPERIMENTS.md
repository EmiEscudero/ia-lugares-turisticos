# Real-data pilot · October 19, 2026

The selected pretrained EfficientNet-B0 recognizes the correct destination first on **66 / 209 test photos (31.6%)**, and includes it in the top five on **114 / 209 (54.5%)**. The adapted original CNN achieves **4 / 209 (1.9%)** first-choice accuracy on the same labels and split. This supports transfer learning for this collection, while showing that reliable destination identification needs substantially better data.

These are **164-way destination recognition** results. The original notebook's 89.99% / 96.53% figures measure a different, two-category task (beach versus town) and cannot be compared to this table.

## Data audit and scope

All **4,668 Drive photos** were copied to local storage, with expected sizes checked: 1,122,551,577 bytes. Every photo decoded successfully. Drive was used only for reading; no original was edited, moved, renamed, or deleted. A local SHA-256 receipt is saved under `artifacts/source_copy_receipt.json`.

The 487 category/place folders describe **480 canonical state/destination identities**. Global exact, mirrored, and difference-hash grouping found **2,294 image families**. A conservative quarantine excludes **30 families / 120 photos** that span conflicting destination names. This includes shared pictures under Cacaluta variants and different Playa de Oro labels. They need manual geographic and visual review; exclusion does not decide the correct label.

After keeping one representative per nonconflicting family and requiring at least six independent groups per destination, the pilot contains **164 destinations / 1,375 photos**: **957 train, 209 validation, 209 test**. The other **316 destinations** remain in the original copy and appear in the preparation report. This model cannot identify all 480 places.

An independent brute-force audit of all 1,375 retained photos found **zero cross-split exact or mirrored near matches at Hamming distance ≤4**. That verifies the implemented matching criterion, not the absence of every crop, same-source image, or semantic duplicate. Perceptual-hash false matches and geographic naming overlap still need human review. Most classes have only one evaluation photo, so these measurements are noisy.

The detailed image manifest (`data/clean_pilot/manifest.csv`) and preparation report (`data/clean_pilot/preparation_report.json`) are kept locally, together with `data/clean_pilot/metadata.json`, because the image paths can contain private source identifiers. They are not published with the code. The [destination mapping](../experiments/pilot/metadata.json), aggregate measurement reports, and [independent verification](../experiments/pilot/split_verification.json) remain public. Manifest SHA-256: `65ab0d996ed647b427c324b9d2df9b3592385333cc3481d95fc9d7e1a81dc031`.

## Controlled comparisons

All four runs use the same cleaned split, seed 42, batch size 32, four Windows loader workers, AdamW, label smoothing 0.05, training-only image augmentation, CUDA mixed precision, a 35-epoch limit, and patience 8. Every included class has training, validation, and test support. The CNN uses its original 128-pixel input scale and no ImageNet normalization; EfficientNet uses its pretrained 224-pixel preprocessing.

| Run | Best / completed epoch | Validation top-1 | Validation macro-F1 | Test top-1 | Test top-5 | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adapted original CNN | 13 / 21 | 6.2% | 0.029 | 1.9% | 11.5% | 0.014 |
| EfficientNet, initial fine-tuning | 24 / 32 | 25.8% | 0.228 | 27.3% | 48.3% | 0.229 |
| EfficientNet, frozen features | 22 / 30 | 34.4% | 0.287 | 33.5% | 53.6% | 0.266 |
| **EfficientNet, stable fine-tuning (selected)** | **27 / 35** | **34.9%** | **0.310** | **31.6%** | **54.5%** | **0.246** |

The model was selected **before test evaluation**, using maximum validation macro-F1 and validation loss to break ties. Each saved model then received one final test evaluation. The frozen model happens to score higher on test top-1; it was not substituted after seeing the test. Further decisions require new validation evidence, repeated seeds, and ultimately a stronger independent test collection.

EfficientNet head learning rate is 0.001, with backbone rate 0.00001 after three frozen epochs. The frozen-only run keeps its backbone fixed for the entire run. The initial fine-tuning run updates BatchNorm running statistics and enables cuDNN autotuning; these two flags were implicit in that earlier implementation and are reconstructed explicitly in the saved summary. The stable run keeps pretrained BatchNorm statistics fixed and leaves autotuning off. Since two settings changed, this comparison does not isolate the cause of its gain.

The legacy comparison retains the original three convolutional stages and 512-unit dense layer but replaces the hierarchical heads with the same destination head as the other runs. Its optimizer, augmentation, GPU path, and split are shared with the new workflow. It is a **legacy-style baseline**, not an exact rerun of the old notebook. No historical trained weights were available.

## RTX 4060 measurements

Machine: NVIDIA GeForce RTX 4060, 8,188 MiB VRAM, driver 616.92; Python 3.13.2, PyTorch 2.11.0+cu128, TorchVision 0.26.0+cu128, Pillow 12.3.0, Streamlit 1.66.0. Epoch time includes train and validation loading/computation but excludes checkpoint serialization and environment/weight downloads. The median below uses epochs 5 onward.

| Run | First epoch | Median subsequent epoch | Peak allocated VRAM | Sum of recorded epoch time |
| --- | ---: | ---: | ---: | ---: |
| Adapted original CNN | 38.3 s | 5.6 s | 236 MiB | 152 s |
| Initial fine-tuning | 88.9 s | 12.9 s | 1,471 MiB | 536 s |
| Frozen features | 55.3 s | 6.4 s | 215 MiB | 245 s |
| Stable fine-tuning | 53.2 s | 6.6 s | 1,467 MiB | 289 s |

The initial autotuned run also spends 74.2 s in the first unfrozen epoch; the stable run spends 16.6 s there. Windows worker startup contributes to first-epoch overhead. Measurements came from sequential runs on this desktop, with occasional CPU UI testing; GPU allocated memory excludes driver reservations and other processes. They establish that batch 32 fits comfortably on this machine. The old laptop was not benchmarked, so there is no defensible laptop-versus-desktop speedup factor.

## Reproduce and inspect

[The saved summary](../experiments/pilot/summary.json) includes all metrics, settings, selected epoch, checkpoint digest, and timing. Each run's directory contains its history, test metrics, per-class recall, and confusion counts. Weights and source photos stay local and are excluded from Git.

```powershell
# The downloaded local copy uses Drive file IDs; this inventory restores original-name preference.
.\.venv\Scripts\python.exe -m tourist_places.prepare --source data/raw --output data/new_pilot --min-images 6 --hash-distance 4 --exclude-conflicts --inventory data/drive_inventory.json

# Reuse the exact recorded split, providing the matching source copy.
.\.venv\Scripts\python.exe -m tourist_places.train --prepared data/clean_pilot --data-root data/raw --output runs/reproduction --device cuda --batch-size 32 --epochs 35 --workers 4 --patience 8
```

The original-name inventory, detailed manifest, preparation report, and photo copy remain local. Reproducing the exact recorded split requires authorized access to those local files under `data/clean_pilot/` and the matching source copy. They are not supplied by cloning this repository. For a new authorized collection with original filenames, omit `--inventory`, prepare a new split, and train with its output folder (for example, `--prepared data/new_pilot`). Different mirror representative choices can change bytes; the new split should not be presented as the recorded pilot. Seeds do not promise bit-for-bit CUDA reproducibility. `runs/recommended/best.pt` locally contains the selected stable model; SHA-256 `3fcf631a8645d2bb4f2d8b790683f67192993443eee270dc38094ea2e3c9ee1e`.

## Interface and next experiments

The local Streamlit interface defaults to Spanish and has an English switch. Shareable `?lang=es` / `?lang=en` URLs select the initial language. Photo selection and ranked scores persist when the language changes. Uploads are processed in memory, with server-side size, format, resolution, and decoding checks. CPU inference leaves the GPU available for training. The app's score is **not calibrated confidence** or an unknown-place detector; its minimum-score control is a display heuristic.

The three optional UI examples come from the new **validation** split, never from the held-out test set. Do not interpret individual demo scores as a benchmark. Image provenance is recorded locally in `data/ui_example_provenance.json`.

Next priorities: manually resolve destination boundaries and conflicting families; collect independent photos for excluded classes; group by source/photographer; repeat several seeds; compare pretrained embedding retrieval; and validate rejection/calibration using unknown destinations. New optimization should use validation data, not these now-reported test results. Before a public portfolio deployment, package the chosen checkpoint, check photo usage rights, and measure behavior on a new collection of real visitor uploads.
