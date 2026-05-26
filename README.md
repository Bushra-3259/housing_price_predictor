# 🏡 California Housing Price Predictor (End-to-End Tuned LightGBM Pipeline)

A production-grade Machine Learning application deployed via Streamlit that predicts the median house value of neighborhoods in California. This project significantly improves upon standard baseline tutorial models by incorporating target un-biasing, advanced spatial feature engineering, and robust cross-validation tracking.

---

## 🚀 Key Engineering Enhancements (Over the Baseline Tutorial)

Unlike standard implementations that copy basic notebook tutorials, this project introduces major pipeline upgrades:

* **Mitigated Capping Bias:** Audited the dataset to remove artificial target truncation limits (capped at $500,001), preventing decision trees from training on bottlenecked values and reducing model errors.
* **Advanced Spatial Feature Engineering:** Engineered proximity variables calculating the direct distance from each neighborhood block to California's primary economic hubs (San Francisco and Los Angeles).
* **State-of-the-Art Architecture:** Swapped standard boosting models for **Microsoft's LightGBM Regressor**, utilizing histogram-based binning to maximize training speed and accuracy.
* **Automated Target Scaling:** Integrated a `TransformedTargetRegressor` wrapper to automatically handle mathematical log-transformations (`np.log1p`) to fix right-skewed price distributions, reversing it seamlessly during live cloud inference.
* **Robust Spatial Cross-Validation:** Implemented a structural `GroupKFold` validator based on coordinate blocks to evaluate real-world model generalizability on completely unseen geographic regions.

---

## 🛠️ The Tech Stack

* **Language:** Python 3.10
* **Environment:** Google Colab (Cloud Architecture)
* **Core Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy, Joblib
* **Deployment:** Streamlit Community Cloud

---

## 📊 Performance Metrics

| Evaluation Strategy | Metric Used | Model Performance |
| :--- | :--- | :--- |
| **Random K-Fold CV** | Mean RMSE | *[Insert your Best CV RMSE Value from your GridSearch here]* |
| **Spatial Group K-Fold CV** | Mean RMSE | *[Insert your Robust Spatial CV RMSE here]* |

---

## 💻 How to Use the App

1. Enter the target neighborhood metrics (Coordinates, Median Income, Total Rooms, Households, etc.).
2. Select the relative **Ocean Proximity** classification.
3. Click **Calculate Estimated Value** to trigger live pipeline inference and drop an un-skewed real estate dollar evaluation on screen.
