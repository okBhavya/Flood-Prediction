# Flood Prediction Using Multi-Temporal Sentinel-1 SAR & U-Net / Attention U-Net (Team 26)

End-to-end Sentinel-1 SAR flood mapping and deep-learning semantic segmentation pipeline built for **Team 26** (Madhya Pradesh — **Chhatarpur** & **Satna** districts, plus the common **Dataset-I** benchmark).

---

## Repository Structure

```text
Team_26/
├── 2026_Flood_Project/                  # Task-II preprocessing, unsupervised flood masks & report
│   └── Madhya_Pradesh/
│       ├── Code/                        # Flood_Data_Preprocessing_speckle_filter.ipynb, T1.ipynb - T3.ipynb
│       ├── Processed data/Chhatarpur/
│       │   └── Mask/                    # Cleaned_flood_mask.tif & intermediate Task-II masks
│       └── Report/                      # Team_26_Task-II_Report.pdf & figures/
├── Chhatarpur/
│   └── Mask/                            # District-level Cleaned_flood_mask.tif (EPSG:32644)
├── Dataset-I/                           # Common 4-band SAR flood benchmark (64x64 patches)
│   ├── Codes/                           # Preparing_for_Input.ipynb, make_patches_updated.py
│   ├── train/                           # 3,646 patch groups (before_VV/VH, after_VV/VH, mask)
│   └── val/                             # 470 patch groups (before_VV/VH, after_VV/VH, mask)
├── Task III_Chattarpur/                 # Task-III primary deliverable — Chhatarpur District
│   ├── Codes/
│   │   ├── Model_1_UNet.ipynb           # Model 1 (U-Net on Dataset-I)
│   │   ├── Model_1_weights.pth          # Best Model 1 checkpoint (Epoch 49)
│   │   ├── Model_2_District_Chhatarpur.ipynb  # Model 2 (Attention U-Net on Chhatarpur Dataset_II)
│   │   ├── Model_2_Chhatarpur_weights.pth     # Best Model 2 checkpoint (Epoch 24)
│   │   └── make_patches_updated.py
│   ├── Dataset_II/                      # 4,531 aligned 64x64 SAR patch groups + patch_index.csv
│   ├── Outputs/
│   │   ├── Model_1/                     # history.csv, results.json, loss_curve.png, collage.png, test_predictions/
│   │   ├── Model_2/                     # history.csv, results.json, loss_curve.png, collage.png, test_predictions/
│   │   └── figures/                     # Architecture schematic, equation renders, enhanced curves & district maps
│   └── Report/                          # Task_III_Report.pdf
└── Task III/                            # Task-III Satna District deliverables
    ├── Codes/                           # Model_1_UNet.ipynb, Model_2_District.ipynb, .pth checkpoints
    ├── Dataset_II/                      # patch_index.csv (11,669 aligned patch groups)
    ├── Outputs/                         # Model_1/, Model_2/, figures/
    └── Report/                          # Task_III_Report.pdf
```

> **Note on Large Full-Scene GeoTIFFs:** Raw/full-scene Sentinel-1 `Gamma0_*.tif` and `*_dB.tif` rasters (`~1.74 - 1.87 GiB` per polarization band) are excluded via [`.gitignore`](.gitignore) because individual files exceed GitHub's `100 MB` file size limit. All derived `64x64` patch datasets (`Dataset-I`, Chhatarpur `Dataset_II`), cleaned binary flood masks (`Cleaned_flood_mask.tif`), trained `.pth` weights, notebooks with saved outputs, evaluation figures, and PDF reports are included directly in this repository.

---

## Models & Verified Metrics

All models consume 4-channel multi-temporal Sentinel-1 SAR inputs `[before_VV, before_VH, after_VV, after_VH]` with per-band normalization and spatial-block train/validation splitting.

| Model | Dataset / District | Architecture | Best Epoch | Val IoU | Val Dice | Test IoU | Test Dice |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | Common `Dataset-I` (`3,464` train / `447` val patches kept) | `UNet` (`in_channels=4`, `base=32`) | 49 / 50 | `0.8829` | `0.9378` | `0.8789` | `0.9355` |
| **Model 2 (Chhatarpur)** | `Chhatarpur Dataset_II` (`3,722` train / `809` val patches) | `AttentionUNet` (`in_channels=4`, `base=32`) | 24 / 25 | `0.8999` | `0.9473` | `0.9101` | `0.9529` |
| **Model 2 (Satna)** | `Satna Dataset_II` (`9,413` train / `1,994` val patches kept) | `AttentionUNet` (`in_channels=4`, `base=32`) | 37 / 40 | `0.8378` | `0.9117` | `0.8308` | `0.9076` |

### Key Highlights (Chhatarpur V2 Pipeline)
- **SNAP 14.0.0 Calibration & Terrain Correction (`EPSG:32644`, `10 m`):** Pre-flood (`2026-05-23`) and post-flood (`2026-09-03`) Sentinel-1 GRD scenes processed with `Apply-Orbit-File` -> `ThermalNoiseRemoval` -> `Calibration` (`Gamma0`) -> `Refined Lee` speckle filter -> `Terrain-Correction` -> `Collocate` on a common `28,535 x 21,578` grid.
- **Zero Spurious Splits or Nodata:** All `4,531` Chhatarpur `64x64` patch groups in `Task III_Chattarpur/Dataset_II/` pass nodata (`0` skipped) and flood-fraction (`0` skipped) validation checks, producing the stitched district prediction raster `Chhatarpur_Before20260523_After20260903_mask.tif`.

---

## Reproducing Evaluation / Inference

1. Install Python dependencies (PyTorch, `rasterio`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `tqdm`):
   ```bash
   pip install torch torchvision rasterio numpy pandas matplotlib scikit-learn scipy tqdm jupyter
   ```
2. Open any notebook under `Team_26/Task III_Chattarpur/Codes/` or `Team_26/Task III/Codes/`:
   - `Model_1_UNet.ipynb` runs Model 1 training / evaluation on `Team_26/Dataset-I/`.
   - `Model_2_District_Chhatarpur.ipynb` runs Model 2 training, validation evaluation, and full-district stitched map inference on `Team_26/Task III_Chattarpur/Dataset_II/` using `Model_2_Chhatarpur_weights.pth`.
3. To inspect saved metrics or pre-computed outputs directly without re-training, refer to:
   - [`Team_26/Task III_Chattarpur/Report/Task_III_Report.pdf`](Team_26/Task%20III_Chattarpur/Report/Task_III_Report.pdf)
   - [`Team_26/Task III_Chattarpur/Outputs/Model_2/results.json`](Team_26/Task%20III_Chattarpur/Outputs/Model_2/results.json)
   - [`Team_26/Task III/Report/Task_III_Report.pdf`](Team_26/Task%20III/Report/Task_III_Report.pdf)
