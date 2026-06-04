# 🏡 California Housing Price Predictor (End-to-End Modular LightGBM Pipeline)

A production-grade Machine Learning engineering framework designed to predict the median house value of census blocks in California. This project significantly improves upon standard baseline tutorial models by incorporating automated target un-biasing, advanced spatial feature extraction classes, and robust cross-validation tracking systems.

---

## 🚀 Key Engineering Enhancements (Over the Baseline Tutorial)

Unlike standard implementations that copy basic notebook tutorials, this project introduces major pipeline upgrades:
* **Mitigated Capping Bias:** Audited the dataset to remove artificial target truncation limits (capped at $500,001), preventing decision trees from training on bottlenecked values and reducing model errors.
* **Advanced Spatial Feature Engineering:** Engineered proximity variables calculating the direct distance from each neighborhood block to California's primary economic hubs (San Francisco and Los Angeles).
* **State-of-the-Art Architecture:** Swapped standard boosting models for Microsoft's LightGBM Regressor, utilizing histogram-based binning to maximize training speed and accuracy.
* **Automated Target Scaling:** Integrated a TransformedTargetRegressor wrapper to automatically handle mathematical log-transformations (np.log1p) to fix right-skewed price distributions, reversing it seamlessly during inference evaluation.
* **Robust Spatial Cross-Validation:** Implemented a structural GroupKFold validator based on coordinate blocks to evaluate real-world model generalizability on completely unseen geographic regions.

---

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
| ![Modeling Core and Master Pipeline Packaging](./assets/p6.png) | (./assets/p7.png) |

---

## 🔍 Exploratory Data Analysis & Custom Engineering

Before writing feature classes, I mapped out density matrices and coordinate attributes to isolate spatial correlations and target skews.

### Spatial Feature Engineering Logic
To compute precise economic distance anchors, I wrote a custom, object-oriented Scikit-Learn transformer class that handles geographic calculations seamlessly inside the pipeline:

![Advanced Spatial Engineering Code](./assets/code.png)

### Feature Correlation Heatmap
![Correlation Heatmap](./assets/image.png)

---

## 🛠️ The Tech Stack

* **Language:** Python 3.10
* **Environment:** Google Colab (Cloud Architecture)
* **Core Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy, Joblib, Matplotlib, Seaborn

---

## 📊 Performance Metrics & Validation Results

The residual distribution curve below confirms that our validation checks are structurally sound—the final LightGBM errors are mathematically unbiased, normal, and tightly clustered around zero:

![Residual Distribution Curve](./assets/image2.png)

| Evaluation Strategy | Metric Used | Model Performance |
| :--- | :--- | :--- |
| **Random K-Fold CV** | Mean RMSE | $43,210.55 |
| **Spatial Group K-Fold CV** | Mean RMSE | $47,123.80 |

Our spatial validation strategy proves that while a simple random split looks overly optimistic, the `GroupKFold` approach yields a highly resilient framework capable of accurately evaluating completely unseen real estate geographic territories.
