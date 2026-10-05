# Controlled detector-settings sensitivity

Preset settings were compared on identical image cohorts, changing confidence or input canvas within one detector at a time. Weights, NMS and the exactly-one-face rule were held fixed. Each configuration used its own detector instance; execution order was randomised per image. Identical-setting repeat controls quantify observed run variation. The plan and cohort fingerprints were saved before detector evaluation; all planned settings are reported, with no winner selected.

This is an exploratory intervention on benchmark processing, designed after the original results were inspected. BFW development/test cohorts are identity-disjoint; these are existing benchmark partitions, not new independent confirmation. LFW and CPLFW are not independent populations. Cohorts sample identities with at least two eligible images; they are not the full pair protocols or representative deployment samples.

**Denominators:** every intended image, including decode failures. One detected face means eligibility for recognition, not correct identity or successful embedding. Unprocessed images per 1,000 are unresolved workload, not false duplicate referrals. Review time and the correctness of extra detections were not annotated. No recognition threshold, main model or gallery was changed.

**Cost:** warm wall-clock detection calls, including resizing, inference and NMS, excluding decoding, recognition and gallery search. Model bytes are identical across settings of one detector. Times describe this run and machine; they are not a latency confidence interval.

**Uncertainty:** 2000 paired percentile identity-bootstrap replicates, stratified by cohort group. The same identity draws are used across settings. Intervals are exploratory and unadjusted for multiple comparisons, conditional on these weights, cohorts and settings. A zero repeat-control difference cannot prove numerical determinism.

## BFW_development

256 images from 128 identities.

| Setting | Confidence | Canvas | Zero | One | Multiple | Decode failure | Unprocessed / 1,000 | Detection mean / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCRFD_baseline | 0.5 | 320 | 1 | 254 | 1 | 0 | 7.8 | 26.16 / 30.34 |
| SCRFD_canvas_160 | 0.5 | 160 | 0 | 255 | 1 | 0 | 3.9 | 9.49 / 12.89 |
| SCRFD_canvas_640 | 0.5 | 640 | 254 | 2 | 0 | 0 | 992.2 | 93.35 / 100.03 |
| SCRFD_confidence_0.3 | 0.3 | 320 | 0 | 254 | 2 | 0 | 7.8 | 25.99 / 29.60 |
| SCRFD_confidence_0.7 | 0.7 | 320 | 3 | 253 | 0 | 0 | 11.7 | 26.20 / 30.11 |
| SCRFD_confidence_0.9 | 0.9 | 320 | 251 | 5 | 0 | 0 | 980.5 | 25.97 / 30.35 |
| SCRFD_repeat | 0.5 | 320 | 1 | 254 | 1 | 0 | 7.8 | 25.95 / 28.90 |
| YuNet_baseline | 0.9 | native | 10 | 246 | 0 | 0 | 39.1 | 2.50 / 6.39 |
| YuNet_canvas_320 | 0.9 | 320 | 1 | 254 | 1 | 0 | 7.8 | 5.00 / 5.49 |
| YuNet_canvas_640 | 0.9 | 640 | 175 | 81 | 0 | 0 | 683.6 | 19.86 / 21.45 |
| YuNet_confidence_0.7 | 0.7 | native | 0 | 256 | 0 | 0 | 0.0 | 2.51 / 6.18 |
| YuNet_confidence_0.8 | 0.8 | native | 0 | 256 | 0 | 0 | 0.0 | 2.50 / 6.09 |
| YuNet_confidence_0.95 | 0.95 | native | 250 | 6 | 0 | 0 | 976.6 | 2.49 / 6.42 |
| YuNet_repeat | 0.9 | native | 10 | 246 | 0 | 0 | 39.1 | 2.53 / 6.62 |

Changes below are variant minus the same detector's baseline. Transitions describe the same images, not different surviving subsets.

| Variant | One-face coverage change, percentage points (95% CI) | Zero → one | One → multiple | One → zero | Changed outcomes |
| --- | --- | --- | --- | --- | --- |
| YuNet_confidence_0.7 | +3.91 [+1.95, +6.25] | 10 | 0 | 0 | 10 |
| YuNet_confidence_0.8 | +3.91 [+1.95, +6.25] | 10 | 0 | 0 | 10 |
| YuNet_confidence_0.95 | -93.75 [-96.48, -90.62] | 0 | 0 | 240 | 240 |
| YuNet_canvas_320 | +3.12 [+1.17, +5.08] | 8 | 0 | 0 | 9 |
| YuNet_canvas_640 | -64.45 [-69.53, -59.38] | 0 | 0 | 165 | 165 |
| YuNet_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |
| SCRFD_confidence_0.3 | +0.00 [-1.17, +0.78] | 1 | 1 | 0 | 2 |
| SCRFD_confidence_0.7 | -0.39 [-1.56, +0.78] | 0 | 0 | 2 | 3 |
| SCRFD_confidence_0.9 | -97.27 [-99.22, -95.31] | 0 | 0 | 249 | 250 |
| SCRFD_canvas_160 | +0.39 [+0.00, +1.17] | 1 | 0 | 0 | 1 |
| SCRFD_canvas_640 | -98.44 [-100.00, -96.48] | 0 | 0 | 253 | 254 |
| SCRFD_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |

### What the interventions changed

Changing YuNet confidence from 0.9 to 0.8 changed zero detections from 10 to 0 and multiple detections from 0 to 0. The net one-face coverage change was +3.91 percentage points (95% CI +1.95 to +6.25; excludes zero). Unresolved images changed from 39.1 to 0.0 per 1,000 intended images.

Changing YuNet canvas from native to 320 changed zero detections from 10 to 1 and multiple detections from 0 to 1. The net one-face coverage change was +3.12 percentage points (95% CI +1.17 to +5.08; excludes zero). Unresolved images changed from 39.1 to 7.8 per 1,000 intended images.

Changing SCRFD confidence from 0.5 to 0.7 changed zero detections from 1 to 3 and multiple detections from 1 to 0. The net one-face coverage change was -0.39 percentage points (95% CI -1.56 to +0.78; includes zero). Unresolved images changed from 7.8 to 11.7 per 1,000 intended images.

Changing SCRFD canvas from 320 to 640 changed zero detections from 1 to 254 and multiple detections from 1 to 0. The net one-face coverage change was -98.44 percentage points (95% CI -100.00 to -96.48; excludes zero). Unresolved images changed from 7.8 to 992.2 per 1,000 intended images.

## BFW_test

256 images from 128 identities.

| Setting | Confidence | Canvas | Zero | One | Multiple | Decode failure | Unprocessed / 1,000 | Detection mean / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCRFD_baseline | 0.5 | 320 | 0 | 256 | 0 | 0 | 0.0 | 25.50 / 28.46 |
| SCRFD_canvas_160 | 0.5 | 160 | 0 | 256 | 0 | 0 | 0.0 | 8.84 / 11.06 |
| SCRFD_canvas_640 | 0.5 | 640 | 242 | 14 | 0 | 0 | 945.3 | 91.96 / 98.72 |
| SCRFD_confidence_0.3 | 0.3 | 320 | 0 | 254 | 2 | 0 | 7.8 | 25.48 / 29.37 |
| SCRFD_confidence_0.7 | 0.7 | 320 | 6 | 250 | 0 | 0 | 23.4 | 25.32 / 28.67 |
| SCRFD_confidence_0.9 | 0.9 | 320 | 254 | 2 | 0 | 0 | 992.2 | 25.39 / 28.66 |
| SCRFD_repeat | 0.5 | 320 | 0 | 256 | 0 | 0 | 0.0 | 25.61 / 28.95 |
| YuNet_baseline | 0.9 | native | 10 | 246 | 0 | 0 | 39.1 | 2.50 / 6.20 |
| YuNet_canvas_320 | 0.9 | 320 | 3 | 253 | 0 | 0 | 11.7 | 5.06 / 5.69 |
| YuNet_canvas_640 | 0.9 | 640 | 153 | 103 | 0 | 0 | 597.7 | 19.64 / 21.01 |
| YuNet_confidence_0.7 | 0.7 | native | 0 | 256 | 0 | 0 | 0.0 | 2.50 / 6.23 |
| YuNet_confidence_0.8 | 0.8 | native | 1 | 255 | 0 | 0 | 3.9 | 2.48 / 5.87 |
| YuNet_confidence_0.95 | 0.95 | native | 244 | 12 | 0 | 0 | 953.1 | 2.52 / 6.51 |
| YuNet_repeat | 0.9 | native | 10 | 246 | 0 | 0 | 39.1 | 2.51 / 5.88 |

Changes below are variant minus the same detector's baseline. Transitions describe the same images, not different surviving subsets.

| Variant | One-face coverage change, percentage points (95% CI) | Zero → one | One → multiple | One → zero | Changed outcomes |
| --- | --- | --- | --- | --- | --- |
| YuNet_confidence_0.7 | +3.91 [+1.95, +6.64] | 10 | 0 | 0 | 10 |
| YuNet_confidence_0.8 | +3.52 [+1.56, +5.86] | 9 | 0 | 0 | 9 |
| YuNet_confidence_0.95 | -91.41 [-94.53, -87.89] | 0 | 0 | 234 | 234 |
| YuNet_canvas_320 | +2.73 [+0.78, +5.08] | 7 | 0 | 0 | 7 |
| YuNet_canvas_640 | -55.86 [-61.72, -50.00] | 1 | 0 | 144 | 145 |
| YuNet_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |
| SCRFD_confidence_0.3 | -0.78 [-1.95, +0.00] | 0 | 2 | 0 | 2 |
| SCRFD_confidence_0.7 | -2.34 [-4.69, -0.39] | 0 | 0 | 6 | 6 |
| SCRFD_confidence_0.9 | -99.22 [-100.00, -98.05] | 0 | 0 | 254 | 254 |
| SCRFD_canvas_160 | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |
| SCRFD_canvas_640 | -94.53 [-97.27, -91.80] | 0 | 0 | 242 | 242 |
| SCRFD_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |

### What the interventions changed

Changing YuNet confidence from 0.9 to 0.8 changed zero detections from 10 to 1 and multiple detections from 0 to 0. The net one-face coverage change was +3.52 percentage points (95% CI +1.56 to +5.86; excludes zero). Unresolved images changed from 39.1 to 3.9 per 1,000 intended images.

Changing YuNet canvas from native to 320 changed zero detections from 10 to 3 and multiple detections from 0 to 0. The net one-face coverage change was +2.73 percentage points (95% CI +0.78 to +5.08; excludes zero). Unresolved images changed from 39.1 to 11.7 per 1,000 intended images.

Changing SCRFD confidence from 0.5 to 0.7 changed zero detections from 0 to 6 and multiple detections from 0 to 0. The net one-face coverage change was -2.34 percentage points (95% CI -4.69 to -0.39; excludes zero). Unresolved images changed from 0.0 to 23.4 per 1,000 intended images.

Changing SCRFD canvas from 320 to 640 changed zero detections from 0 to 242 and multiple detections from 0 to 0. The net one-face coverage change was -94.53 percentage points (95% CI -97.27 to -91.80; excludes zero). Unresolved images changed from 0.0 to 945.3 per 1,000 intended images.

## CPLFW_raw

256 images from 128 identities.

| Setting | Confidence | Canvas | Zero | One | Multiple | Decode failure | Unprocessed / 1,000 | Detection mean / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCRFD_baseline | 0.5 | 320 | 1 | 235 | 20 | 0 | 82.0 | 25.90 / 30.60 |
| SCRFD_canvas_160 | 0.5 | 160 | 2 | 235 | 19 | 0 | 82.0 | 9.30 / 12.25 |
| SCRFD_canvas_640 | 0.5 | 640 | 4 | 232 | 20 | 0 | 93.8 | 93.28 / 106.81 |
| SCRFD_confidence_0.3 | 0.3 | 320 | 1 | 224 | 31 | 0 | 125.0 | 26.94 / 30.70 |
| SCRFD_confidence_0.7 | 0.7 | 320 | 58 | 192 | 6 | 0 | 250.0 | 25.87 / 31.68 |
| SCRFD_confidence_0.9 | 0.9 | 320 | 256 | 0 | 0 | 0 | 1000.0 | 26.62 / 31.48 |
| SCRFD_repeat | 0.5 | 320 | 1 | 235 | 20 | 0 | 82.0 | 25.87 / 29.92 |
| YuNet_baseline | 0.9 | native | 70 | 184 | 2 | 0 | 281.2 | 3.29 / 3.68 |
| YuNet_canvas_320 | 0.9 | 320 | 48 | 205 | 3 | 0 | 199.2 | 5.11 / 5.55 |
| YuNet_canvas_640 | 0.9 | 640 | 68 | 184 | 4 | 0 | 281.2 | 19.59 / 21.47 |
| YuNet_confidence_0.7 | 0.7 | native | 6 | 235 | 15 | 0 | 82.0 | 3.28 / 3.60 |
| YuNet_confidence_0.8 | 0.8 | native | 9 | 239 | 8 | 0 | 66.4 | 3.32 / 3.64 |
| YuNet_confidence_0.95 | 0.95 | native | 255 | 1 | 0 | 0 | 996.1 | 3.25 / 3.64 |
| YuNet_repeat | 0.9 | native | 70 | 184 | 2 | 0 | 281.2 | 3.25 / 3.60 |

Changes below are variant minus the same detector's baseline. Transitions describe the same images, not different surviving subsets.

| Variant | One-face coverage change, percentage points (95% CI) | Zero → one | One → multiple | One → zero | Changed outcomes |
| --- | --- | --- | --- | --- | --- |
| YuNet_confidence_0.7 | +19.92 [+14.44, +25.78] | 58 | 7 | 0 | 71 |
| YuNet_confidence_0.8 | +21.48 [+15.62, +26.95] | 60 | 5 | 0 | 66 |
| YuNet_confidence_0.95 | -71.48 [-76.96, -65.62] | 0 | 0 | 183 | 185 |
| YuNet_canvas_320 | +8.20 [+4.30, +12.11] | 25 | 1 | 3 | 29 |
| YuNet_canvas_640 | +0.00 [-4.69, +4.69] | 19 | 2 | 17 | 38 |
| YuNet_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |
| SCRFD_confidence_0.3 | -4.30 [-7.03, -1.95] | 0 | 11 | 0 | 11 |
| SCRFD_confidence_0.7 | -16.80 [-22.27, -10.94] | 0 | 0 | 54 | 68 |
| SCRFD_confidence_0.9 | -91.80 [-94.92, -88.28] | 0 | 0 | 235 | 255 |
| SCRFD_canvas_160 | +0.00 [-1.95, +1.95] | 0 | 2 | 1 | 6 |
| SCRFD_canvas_640 | -1.17 [-3.52, +0.78] | 0 | 4 | 3 | 11 |
| SCRFD_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |

### What the interventions changed

Changing YuNet confidence from 0.9 to 0.8 changed zero detections from 70 to 9 and multiple detections from 2 to 8. The net one-face coverage change was +21.48 percentage points (95% CI +15.62 to +26.95; excludes zero). Unresolved images changed from 281.2 to 66.4 per 1,000 intended images.

Changing YuNet canvas from native to 320 changed zero detections from 70 to 48 and multiple detections from 2 to 3. The net one-face coverage change was +8.20 percentage points (95% CI +4.30 to +12.11; excludes zero). Unresolved images changed from 281.2 to 199.2 per 1,000 intended images.

Changing SCRFD confidence from 0.5 to 0.7 changed zero detections from 1 to 58 and multiple detections from 20 to 6. The net one-face coverage change was -16.80 percentage points (95% CI -22.27 to -10.94; excludes zero). Unresolved images changed from 82.0 to 250.0 per 1,000 intended images.

Changing SCRFD canvas from 320 to 640 changed zero detections from 1 to 4 and multiple detections from 20 to 20. The net one-face coverage change was -1.17 percentage points (95% CI -3.52 to +0.78; includes zero). Unresolved images changed from 82.0 to 93.8 per 1,000 intended images.

## LFW_development

256 images from 128 identities.

| Setting | Confidence | Canvas | Zero | One | Multiple | Decode failure | Unprocessed / 1,000 | Detection mean / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCRFD_baseline | 0.5 | 320 | 0 | 203 | 53 | 0 | 207.0 | 25.43 / 27.66 |
| SCRFD_canvas_160 | 0.5 | 160 | 0 | 206 | 50 | 0 | 195.3 | 9.00 / 10.94 |
| SCRFD_canvas_640 | 0.5 | 640 | 3 | 202 | 51 | 0 | 210.9 | 91.27 / 93.33 |
| SCRFD_confidence_0.3 | 0.3 | 320 | 0 | 193 | 63 | 0 | 246.1 | 25.12 / 26.95 |
| SCRFD_confidence_0.7 | 0.7 | 320 | 3 | 226 | 27 | 0 | 117.2 | 25.43 / 27.50 |
| SCRFD_confidence_0.9 | 0.9 | 320 | 256 | 0 | 0 | 0 | 1000.0 | 25.08 / 26.85 |
| SCRFD_repeat | 0.5 | 320 | 0 | 203 | 53 | 0 | 207.0 | 25.33 / 27.56 |
| YuNet_baseline | 0.9 | native | 0 | 242 | 14 | 0 | 54.7 | 3.21 / 3.56 |
| YuNet_canvas_320 | 0.9 | 320 | 0 | 241 | 15 | 0 | 58.6 | 4.95 / 5.40 |
| YuNet_canvas_640 | 0.9 | 640 | 0 | 232 | 24 | 0 | 93.8 | 19.38 / 20.30 |
| YuNet_confidence_0.7 | 0.7 | native | 0 | 210 | 46 | 0 | 179.7 | 3.22 / 3.56 |
| YuNet_confidence_0.8 | 0.8 | native | 0 | 219 | 37 | 0 | 144.5 | 3.24 / 3.61 |
| YuNet_confidence_0.95 | 0.95 | native | 254 | 2 | 0 | 0 | 992.2 | 3.21 / 3.51 |
| YuNet_repeat | 0.9 | native | 0 | 242 | 14 | 0 | 54.7 | 3.22 / 3.57 |

Changes below are variant minus the same detector's baseline. Transitions describe the same images, not different surviving subsets.

| Variant | One-face coverage change, percentage points (95% CI) | Zero → one | One → multiple | One → zero | Changed outcomes |
| --- | --- | --- | --- | --- | --- |
| YuNet_confidence_0.7 | -12.50 [-16.80, -8.59] | 0 | 32 | 0 | 32 |
| YuNet_confidence_0.8 | -8.98 [-12.89, -5.47] | 0 | 23 | 0 | 23 |
| YuNet_confidence_0.95 | -93.75 [-96.48, -90.62] | 0 | 0 | 240 | 254 |
| YuNet_canvas_320 | -0.39 [-1.56, +0.78] | 0 | 2 | 0 | 3 |
| YuNet_canvas_640 | -3.91 [-6.25, -1.56] | 0 | 10 | 0 | 10 |
| YuNet_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |
| SCRFD_confidence_0.3 | -3.91 [-6.64, -1.56] | 0 | 10 | 0 | 10 |
| SCRFD_confidence_0.7 | +8.98 [+5.46, +12.89] | 0 | 0 | 2 | 28 |
| SCRFD_confidence_0.9 | -79.30 [-83.98, -73.83] | 0 | 0 | 203 | 256 |
| SCRFD_canvas_160 | +1.17 [-1.17, +3.52] | 0 | 3 | 0 | 9 |
| SCRFD_canvas_640 | -0.39 [-3.12, +2.34] | 0 | 3 | 3 | 11 |
| SCRFD_repeat | +0.00 [+0.00, +0.00] | 0 | 0 | 0 | 0 |

### What the interventions changed

Changing YuNet confidence from 0.9 to 0.8 changed zero detections from 0 to 0 and multiple detections from 14 to 37. The net one-face coverage change was -8.98 percentage points (95% CI -12.89 to -5.47; excludes zero). Unresolved images changed from 54.7 to 144.5 per 1,000 intended images.

Changing YuNet canvas from native to 320 changed zero detections from 0 to 0 and multiple detections from 14 to 15. The net one-face coverage change was -0.39 percentage points (95% CI -1.56 to +0.78; includes zero). Unresolved images changed from 54.7 to 58.6 per 1,000 intended images.

Changing SCRFD confidence from 0.5 to 0.7 changed zero detections from 0 to 3 and multiple detections from 53 to 27. The net one-face coverage change was +8.98 percentage points (95% CI +5.46 to +12.89; excludes zero). Unresolved images changed from 207.0 to 117.2 per 1,000 intended images.

Changing SCRFD canvas from 320 to 640 changed zero detections from 0 to 3 and multiple detections from 53 to 51. The net one-face coverage change was -0.39 percentage points (95% CI -3.12 to +2.34; includes zero). Unresolved images changed from 207.0 to 210.9 per 1,000 intended images.

## Interpretation

Within-detector interventions establish how the tested settings change this pipeline's processing outcomes on fixed images. A confidence change can admit a previously rejected face or create a multiple-detection rejection; input resizing changes the detector's effective image scale. Use the measured transition counts and intervals to determine which effect occurred. Do not compare YuNet and SCRFD confidence numbers as a common scale. The study does not identify every cause of the original whole-dataset failure gap, label extra detections as true faces, or establish that higher coverage improves recognition. Any adoption of a changed setting requires development-only recalibration and subsequent evaluation of coverage, duplicate detection, false referrals and total cost.
