## 🛠️ Complete Pipeline Architecture Execution Flow

The full end-to-end processing pipeline is built completely modularly. Here is the sequential execution flow tracing data from raw ingestion to the final validated model output weights:

### ⚙️ Core Preprocessing & Feature Extraction Steps
Below are the architectural layers handling data scrubbing, custom transformers, and initial pipeline structuring:

| Stage 01: Preprocessing Core | Stage 02: Structural Split Core | Stage 03: Feature Processing Switching |
| :---: | :---: | :---: |
| ![Preprocessing](./assets/p1.png) | ![Structural Split](./assets/p2.png) | ![Feature Processing](./assets/p3.png) |

| Stage 04: Engineering Matrix | Stage 05: Pipeline Structure Matrix |
| :---: | :---: |
| ![Engineering Matrix](./assets/p4.png) | ![Pipeline Structure](./assets/p5.png) |

### 📊 HistGradientBoostingRegressor Modeling Core & Packaging
Stages 06 and 07 map out the histogram-based boosting engine configuration and its master integration wrapper, designed for clean, leak-proof training and evaluation:

| Stage 06: HistGradientBoosting Engine | Stage 07: Master Estimator Packaging |
| :---: | :---: |
| ![Modeling Core](./assets/p6.png) | ![Master Pipeline Packaging](./assets/p7.png) |
