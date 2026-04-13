# 🌍 CarbonIQ — Personal Carbon Footprint Analyzer

> An ML-powered web app that predicts your daily carbon footprint, explains the key contributors, and delivers personalised, actionable recommendations — built as a capstone project.

---

## 🧩 Problem

Most carbon calculators return a generic number with no personalisation and no clear path forward. People don't know **which specific behaviour** is driving their footprint or **what to change first** for maximum impact.

## 💡 Solution

CarbonIQ uses a **Stacking Ensemble ML model** (Gradient Boosting + Extra Trees → Ridge) trained on real lifestyle behaviour data to:

- **Predict** daily carbon footprint (kg CO₂e) from 9 lifestyle inputs
- **Explain** which category (transport, electricity, waste, digital) contributes most
- **Recommend** targeted, personalised actions ranked by impact
- **Simulate** "what if I change this habit?" outcomes in real time

---

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.471 | 0.630 | 0.948 |
| Extra Trees | 0.406 | 0.566 | 0.958 |
| Gradient Boosting (tuned) | 0.222 | 0.318 | 0.987 |
| **Stacking Ensemble ✅** | **0.220** | **0.302** | **0.988** |

> ⚠️ Note: A naive LR model with engineered emission features scores 0.9998 — this is **data leakage** (the features are near-linear reconstructions of the target). All models above are trained on raw features only for honest evaluation.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| ML | scikit-learn (GradientBoosting, ExtraTrees, StackingRegressor) |
| Data | pandas, numpy |
| App | Streamlit |
| Visualisation | Plotly |
| Model persistence | joblib |

---

## 📂 Project Structure

```
carbon-footprint-project/
│
├── app.py                                  # Streamlit app
├── train_and_save.py                       # Training + model export script
├── requirements.txt
├── README.md
│
├── personal_carbon_footprint_behavior.csv  # Dataset
│
├── model/
│   ├── stacking_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.pkl
│   └── model_metrics.pkl
│
└── notebook/
    └── Carbon_footprint.ipynb
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/carbon-footprint-project.git
cd carbon-footprint-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_and_save.py
```

This will create the `model/` folder with all saved artefacts.

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ✨ App Features

| Feature | Description |
|---|---|
| 🔢 Prediction | Real-time footprint prediction as you adjust sliders |
| 📊 Breakdown | Donut chart showing contribution per emission category |
| 🌏 Comparison | Your footprint vs India average and global average |
| 💡 Recommendations | Top 5 personalised, ranked action items |
| 🔮 What-If Simulator | Change transport / diet / energy → see projected impact |

---

## 📸 Screenshots

> *(Add screenshots of the running app here)*

---

## 📌 Key Decisions & Learnings

- **Leakage detection**: Engineered emission features are near-linear transforms of the target — including them inflates R² to 0.9998. Removed for honest evaluation.
- **Stacking over single model**: Stacking GradientBoosting + ExtraTrees under a Ridge meta-learner reduces variance and gives the best generalisation.
- **Rule-based recommendations on top of ML**: The model predicts the footprint; a transparent rule layer interprets the inputs to generate human-readable, actionable advice.

---

## 👤 Author

Your Name · [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)