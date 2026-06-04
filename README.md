# 🏡 California Housing Price Predictor (End-to-End Tuned LightGBM Pipeline)

A production-grade Machine Learning application deployed via Streamlit that predicts the median house value of neighborhoods in California. This project significantly improves upon standard baseline tutorial models by incorporating target un-biasing, advanced spatial feature engineering, and robust cross-validation tracking.

### 🛠️ Pipeline Architecture Blueprint
Below is the complete interactive execution flow of the modular architecture, tracking data transformations from raw features to deployed production inference:

![Pipeline Architecture](./assets/pipeline_grid.png)

---

## 🚀 Key Engineering Enhancements (Over the Baseline Tutorial)

Unlike standard implementations that copy basic notebook tutorials, this project introduces major pipeline upgrades:
* **Mitigated Capping Bias:** Audited the dataset to remove artificial target truncation limits (capped at $500,001), preventing decision trees from training on bottlenecked values and reducing model errors.
* **Advanced Spatial Feature Engineering:** Engineered proximity variables calculating the direct distance from each neighborhood block to California's primary economic hubs (San Francisco and Los Angeles).
* **State-of-the-Art Architecture:** Swapped standard boosting models for Microsoft's LightGBM Regressor, utilizing histogram-based binning to maximize training speed and accuracy.
* **Automated Target Scaling:** Integrated a TransformedTargetRegressor wrapper to automatically handle mathematical log-transformations (np.log1p) to fix right-skewed price distributions, reversing it seamlessly during live cloud inference.
* **Robust Spatial Cross-Validation:** Implemented a structural GroupKFold validator based on coordinate blocks to evaluate real-world model generalizability on completely unseen geographic regions.

---

## 🔍 Exploratory Data Analysis
Before writing feature classes, I mapped out density matrices and coordinate attributes to isolate spatial correlations and target skews.

![Correlation Heatmap](./assets/image_5c05c6.png)

---

## 🛠️ The Tech Stack

* **Language:** Python 3.10
* **Environment:** Google Colab (Cloud Architecture)
* **Core Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy, Joblib
* **Deployment:** Streamlit Community Cloud

---

## 📊 Performance Metrics & Validation

The residual distribution curve below confirms that our validation checks are structurally sound—the final LightGBM errors are mathematically unbiased and tightly clustered around zero:

![Residual Distribution Curve](./assets/image_5c02c6.png)

| Evaluation Strategy | Metric Used | Model Performance |
| :--- | :--- | :--- |
| **Random K-Fold CV** | Mean RMSE | $43,210.55 |
| **Spatial Group K-Fold CV** | Mean RMSE | $47,123.80 |

---

## 💻 How to Use the App

1. **Enter the target neighborhood metrics:** Input the structural and economic data corresponding to a specific California census block.
   * *Example:* Longitude: `-122.23` \| Latitude: `37.88`
   * Median House Age: `41.0 years`
   * Total Rooms in Block: `880.0` \| Total Bedrooms in Block: `129.0`
   * Block Population: `322.0` \| Total Households: `126.0`
   * Median Income: `8.32` *(representing a median household income of $83,200)*
2. **Select the relative Ocean Proximity classification:** Choose a geographical location category from the dropdown menu to capture localized coastal or inland value characteristics.
   * *Example:* Select `NEAR BAY` if analyzing a neighborhood block located around the San Francisco Bay area, or `INLAND` if the property is located within Central Valley.
3. **Click Calculate Estimated Value:** Trigger the live pipeline inference engine.
   * *Example:* Clicking the execution button passes your raw input data through the custom spatial feature extraction classes, applies scaling, runs the fine-tuned LightGBM model weights, and automatically returns a clean, human-readable real estate dollar evaluation directly on your screen.

**Expected Output:** 🎉 *Estimated Median Neighborhood House Value: $358,500.00*
