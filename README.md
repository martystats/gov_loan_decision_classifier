# Government Loan Decision Classification System

A production-ready Machine Learning system that predicts loan decision outcomes using structured government financial data.

This project demonstrates end-to-end ML engineering — from data preprocessing and model tuning to deployment architecture and Streamlit integration.

---

## Project Overview

This system classifies government loan applications using structured categorical and numerical features.

The objective is to build a robust, reproducible, and deployment-ready classification pipeline that:

- Trains a tuned Gradient Boosting model
- Preserves feature schema integrity
- Stores model metadata for traceability
- Separates artifacts, data configs, and application logic
- Deploys via a Streamlit interface

The final system transitions from experimentation to a production-engineered architecture.

---

## Business Problem

Government financial institutions require accurate and consistent decision systems for:

- Loan approval workflows  
- Policy compliance checks  
- Decision authority categorization  
- Program type classification  

Manual review processes can be inconsistent and slow.

This ML system provides a standardized predictive layer that improves:

- Decision consistency  
- Speed  
- Audit traceability  

---

## Dataset Features

Key input features include:

- `country`
- `decision_authority`
- `policy_type`
- `program`
- `term`
- Engineered numerical features
- Encoded categorical variables

Categorical configurations are stored externally in:

```
data/categories.json
```

to ensure training-inference consistency.

---

## Model Architecture

Final Model:
**GradientBoostingClassifier (Tuned)**

Optimization:
- RandomizedSearchCV
- Cross-validated F1-macro scoring
- Hyperparameter tuning

Artifacts saved:

```
artifacts/
├── gb_tuned_model.pkl
├── feature_columns.pkl
├── tuned_model_metadata.json
```

---

## Model Metadata Tracking

The system automatically stores:

- Model type
- Best hyperparameters
- Cross-validation score
- Number of features
- Training/test sample size
- Python version
- Creation timestamp

This enables:

- Reproducibility
- Deployment traceability
- Version control safety
- Production debugging

---

## Project Structure

```
gov_loan_amount_classifier/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── gb_tuned_model.pkl
│   ├── feature_columns.pkl
│   └── tuned_model_metadata.json
│
├── data/
│   └── categories.json
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

This structure follows real-world ML deployment standards.

---

## Streamlit Deployment

The application loads:

- Model artifact
- Feature schema
- Category configuration

Paths are dynamically resolved using `Path(__file__)` to ensure portability across environments.

To run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Engineering Highlights

- Separation of training and inference logic  
- Externalized category configuration  
- Structured artifact management  
- Metadata persistence for reproducibility  
- Production-safe file handling  
- Scalable folder architecture  

---

## Performance

Model optimized using:

- F1-macro score (to handle class balance)
- Cross-validated tuning
- Structured feature consistency

Final model selected as official deployment candidate after validation on test set.

---

## Future Improvements

- Add prediction confidence score display  
- Add feature importance visualization panel  
- Add model versioning system  
- Containerize with Docker  
- Deploy to Streamlit Cloud  

---

## Author

Martin Ude  
Machine Learning & Data Science Portfolio Project  

---

## License

This project is for educational and portfolio demonstration purposes.
