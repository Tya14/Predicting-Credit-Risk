# Predicting Credit Risk

## Overview
This project explores the use of modern machine learning techniques to predict credit risk, with a focus on both **model performance** and **interpretability**.

Traditional models like logistic regression are widely used in finance, but they struggle with capturing complex, non-linear relationships. This project compares them with more advanced models such as **Random Forest** and **AdaBoost**, while also addressing the "black box" problem using explainability tools.

---

## Objectives
- Evaluate and compare predictive performance of:
  - Logistic Regression  
  - Random Forest  
  - AdaBoost  
- Improve model performance through:
  - Feature engineering  
  - Handling class imbalance  
- Enhance interpretability using:
  - SHAP (Shapley Additive Explanations)  
  - LIME  
- Provide actionable business insights for lending decisions  

---

## Dataset
- **German Credit Dataset** (UCI Machine Learning Repository)
- 1000 observations with 20 features including:
  - Credit history  
  - Loan amount  
  - Duration  
  - Employment status  
  - Age  
  - Housing  

- Target:
  - `Good Credit Risk (70%)`
  - `Bad Credit Risk (30%)`

---

## Exploratory Data Analysis
### Numerical Features
- Identified skewed distributions (e.g., credit amount, loan duration)
- Applied **log transformations** to reduce skewness and improve model stability  

### Categorical Features
- Detected imbalanced categories  
- Combined rare categories based on similar default rates  
- Applied **one-hot encoding**  

### Class Imbalance
- Addressed using **SMOTE (Synthetic Minority Over-sampling Technique)**  

---

## Feature Engineering
- Created:
  - Interaction terms (e.g., numerical × numerical)
  - Ratio features  
  - Squared terms for non-linearity  
- Removed multicollinearity:
  - Correlation threshold: **0.6**
  - Reduced features from **69 → 46**

---

## Models & Methods

### 1. Logistic Regression
- Regularization: L1, L2, Elastic Net  
- Hyperparameter tuning using GridSearchCV  
- Best ROC-AUC: **0.8497**

---

### 2. Random Forest (Best Model)
- Extensive hyperparameter tuning:
  - Number of trees  
  - Depth  
  - Feature selection  
- Best ROC-AUC: **0.9259**

---

### 3. AdaBoost
- Weak learners: Decision Trees  
- Tuned learning rate & estimators  
- ROC-AUC: **0.8701**

---

## 📈 Results Summary

| Model               | ROC-AUC | Precision | Recall |
|--------------------|--------|----------|--------|
| Logistic Regression | 0.8497 | 0.7603 | 0.7929 |
| **Random Forest**   | **0.9259** | **0.8769** | **0.8143** |
| AdaBoost            | 0.8701 | 0.7879 | 0.7429 |

**Random Forest outperformed all models significantly**

---

## Model Explainability

### SHAP Insights

#### Key Predictors Across Models:
- Checking account status  
- Loan duration  
- Credit history  
- Loan purpose (new car)  
- Age  

### Key Takeaways:
- Applicants **without checking accounts** → higher risk  
- **Longer loan durations** → higher default probability  
- Strong **credit history** → lower risk  
- Age showed unexpected patterns in boosting models  

---

## Business Recommendations
- Require stronger financial verification for applicants without bank accounts  
- Introduce **shorter loan tenure incentives**  
- Adjust **interest rates based on loan duration risk**  
- Use **credit history for risk-based pricing**  
- Monitor **fairness in age-related predictions**  

---

## Tech Stack
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- SHAP  

---

## How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/your-repo-name.git

# Install dependencies
pip install -r requirements.txt

# Run notebook or script
```

---

## Key Highlights
- End-to-end ML pipeline (EDA → Feature Engineering → Modeling → Explainability)  
- Strong emphasis on **interpretability**, not just accuracy  
- Real-world financial application with business insights  

---

## Future Improvements
- Incorporate more advanced models (XGBoost, LightGBM)  
- Deploy as a web app (Streamlit / Flask)  
- Perform fairness and bias audits  

---

