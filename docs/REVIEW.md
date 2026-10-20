# Review of the original tourist-place project

Initial review on October 19, 2026. Original Git revision: `193b193fef4210338b3aa0d6897a762664bd8e4c`. The original-code findings combine static inspection of every tracked notebook/script, all three label CSVs, the Drive folder hierarchy, and an image-content sample. Subsequent full-data experiments are documented separately. The original trained checkpoint is not in the repository.

## Main finding

The RTX 4060 can be used by a new GPU training loop, but the first priorities are label consistency, independent evaluation photos, and measuring correct destination recognition. Faster training alone cannot address these issues.

## Verified findings

| Finding | Evidence | Effect / correction |
| --- | --- | --- |
| The original notebooks train on CPU | Neither `model.ipynb` nor `cnn_pytorch.ipynb` moves the model, inputs, or targets to CUDA | Explicit device handling is needed to use the RTX GPU |
| Reported accuracy does not measure destination recognition | `model.ipynb`, code cell 8, calls `calculate_accuracy(main_outputs, category_labels)`; the secondary predictions are not included | The saved final validation accuracy of 0.8999 measures beach versus town. The separate two-class notebook's 96.53% has the same task limitation |
| Very little training evidence per class | `train_labels.csv`: 3,534 rows, 487 category/place classes, median 6 photos per class; 152 classes have at most 3 photos; six classes have one | Transfer learning and more independent photos are higher priorities than a larger randomly initialized network |
| Evaluation is sparse | `val_labels.csv`: 1,134 rows, 486 represented classes, 213 classes with one validation image | Per-place results are highly variable; use macro metrics and collect a stronger benchmark |
| Validation labels disagree with the lookup | The two Taxco validation rows (`pueblo_1174.jpg`, `pueblo_1175.jpg`) have category 1 / subcategory 31. Taxco is 1 / 41; 1 / 31 is Tepotzotlán | Rebuild mappings from folder identity rather than trying to learn these inconsistent IDs |
| Other metadata inconsistencies exist | 112 training rows and 39 validation rows differ from `labels.csv` in name or state text. This includes spelling/spacing variants and a sequence of shifted Baja California Sur IDs | These totals are metadata mismatches, not a verified count of visually mislabeled photos. Review the images before deciding the authoritative identity |
| Category labels overlap | Seven `(state, name)` destinations appear in both categories: Loreto, Isla Aguada, Sayulita, Bacalar, Isla Mujeres, Tulum, and Sisal | The 487 category/place entries describe 480 unique state/name pairs. Predict a canonical place; retain beach/town as nonexclusive metadata |
| Train/validation content overlaps | Two downloaded Tulum image pairs have identical bytes across splits; see below | Rebuild splits after grouping originals and their derived versions |
| The saved training history indicates overfitting | Combined validation loss reaches 4.4416 at epoch 6 and finishes at 6.3532 at epoch 50, while training loss falls to 0.0255 | Save the best model and stop on relevant validation metrics |
| The custom CNN has a large dense layer | `fc1` alone has 8,389,120 parameters, about 97% of the roughly 8.66 million total | Use a pretrained feature extractor and global pooling instead of fitting most parameters from a few examples |
| No held-out test split is supplied | Only train and validation CSVs are tracked | Reserve independent test photos for a final evaluation |

The `val_labels.csv` header also spells `subcategory` as `subcategoty`. The old notebook reads by column position, so this typo alone does **not** explain its behavior. The audit accepts it and flags it; the new manifest has an explicit schema.

## Confirmed duplicate examples

Under `PLAYAS/QUINTANA_ROO/TULUM`:

| Training file | Validation file | SHA-256 (same for both files) |
| --- | --- | --- |
| [tulum-beach2.jpg](https://drive.google.com/file/d/1N5SfNlQnEpht9Vf4dGldQewCTc78R1V6/view) | [tulum-beach-iii2.jpg](https://drive.google.com/file/d/1UqFfpNXzxTW33FzPpsrTA_HX8J36rzSH/view) | `0f2ff7be03c1e35538a1c53767a75266820494162cefc01114543bd66a9b3b84` |
| [espejo (2).jpg](https://drive.google.com/file/d/1wQwugljmUzg9aSL8MYQc-f6c6VqfLmUM/view) | [espejo (14).jpg](https://drive.google.com/file/d/1bFbDtoAMYt43McbATdeYNkrGVhDu4Pqy/view) | `7776ad195669a372d25c4c0c6f119ca8186753e32dc9780ac4cf7a517419cf09` |

All four images are 1600 × 1200 pixels. Visual inspection also confirms that the second pair depicts the mirrored scene from the first pair. The original and mirrored files differ slightly after JPEG encoding, which is why filename checks and byte hashes alone are insufficient. These samples prove leakage in the Drive split; they do not establish the total amount of leakage or exactly which renamed CSV files they became.

## Recommended sequence

1. Audit the entire image collection, resolve conflicting destination identities, group duplicates and derived copies, and save new splits. Review class coverage after deduplication.
2. Use ImageNet-pretrained EfficientNet-B0 at 224 pixels as the first baseline: train the classifier with the backbone frozen, then fine-tune gently. Transfer learning is specifically useful when the target dataset is small; see the [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). This is a candidate to benchmark, not a guaranteed best model.
3. Measure end-to-end destination top-1/top-5, macro-F1, per-place recall, and confusion counts. Keep the held-out test split out of training/model selection.
4. Enable CUDA, mixed precision, and a measured input pipeline. [PyTorch's mixed-precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) describes reduced precision for speed and memory; its [performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) covers worker processes and pinned memory.
5. Collect varied new photographs for underrepresented and visually confusable places. Generic beach scenes may require distinct landmarks or additional location context. Consider retrieval using pretrained image embeddings as a later comparison once a clean baseline exists.

## Implementation boundaries

The new Python package implements this training/evaluation workflow separately from the historical notebooks and GUI. A completed [real-data comparison](EXPERIMENTS.md) measures destination accuracy on the same cleaned split for both a legacy-style CNN and pretrained models. Automatic near-duplicate grouping spans the whole collection, including mirrored views. Conflicting families are quarantined explicitly for this exploratory run. Crops, perceptual-hash false positives, broad geographic labels, and source independence still need review.

The local machine reports an NVIDIA GeForce RTX 4060 with 8,188 MiB available GPU memory and driver 616.92. Initial batch-size recommendations are starting points; measure epoch times and peak memory on the real data before comparing hardware speed.

## Validation of the new workflow

- 27 Python regression checks and four JavaScript component checks passed, including Spanish/English state preservation. Environment: Python 3.13.2, PyTorch 2.11.0+cu128, TorchVision 0.26.0+cu128, Pillow 12.3.0, Streamlit 1.66.0.
- A two-epoch smoke test with official pretrained EfficientNet-B0 weights and 24 synthetic images completed on the RTX 4060, using CUDA mixed precision, two Windows data-loader workers, a frozen-backbone epoch, and an unfrozen fine-tuning epoch.
- The saved checkpoint was reloaded for GPU test-split evaluation and CPU prediction. The low-confidence path returned destination candidates marked uncertain.
- The actual four downloaded Drive sample files were all grouped into one independent scene by the new content/mirror checks.
- All 4,668 source photos were copied locally using read-only Drive access and decoded successfully. The original collection was not changed. The new 164-destination pilot contains 957 training, 209 validation, and 209 test photos after duplicate filtering and explicit conflict quarantine.
- An independent brute-force audit found zero cross-split exact or mirrored near matches at the preparation threshold. This does not establish photographer/source independence or detect every crop.
- Four real-data comparisons completed on the GPU. The model selected by validation macro-F1 achieves 31.6% test top-1 / 54.5% top-5, versus 1.9% / 11.5% for the adapted original CNN on the same destination task and split.

Full settings, limitations, measurements, and reproducible evidence are in [the experiment report](EXPERIMENTS.md). Synthetic smoke-test scores are not recognition results. The next empirical priority is a larger, manually reviewed, source-independent photo collection rather than relying on these small pilot test results for further tuning.
