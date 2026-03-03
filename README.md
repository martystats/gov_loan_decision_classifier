# Government Loan Amount Classification System

## Project Overview
This project builds a machine learning classification system to predict loan amount category (High vs Low).

The final production model is a tuned Gradient Boosting Classifier optimized using RandomizedSearchCV.

---

## Final Model
- Model: GradientBoostingClassifier (Tuned)
- Validation Strategy: Stratified Cross-Validation
- Optimization: RandomizedSearchCV (f1_macro scoring)

---

## Project Structure

gov_loan_amount_classifier/
│
├── app/                 # Streamlit deployment interface
├── artifacts/           # Saved model + schema + metadata
├── data/                # Sample inputs
├── notebooks/           # Model development notebook
├── outputs/             # Prediction exports (ignored by git)
├── requirements.txt
└── .gitignore

---

## Model Artifacts
- gb_tuned_model.pkl
- feature_columns.pkl
- tuned_model_metadata.json

---

## Next Steps
- Build Streamlit interface
- Local deployment
- GitHub upload