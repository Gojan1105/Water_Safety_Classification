# 💧 Water Safety Classification

A machine learning project to classify water samples as **safe** or **unsafe** for consumption based on chemical properties. This project also includes a user-friendly **Streamlit web application** for real-time prediction and visualization.

---

## 📚 Project Details

- **Program:** Naan Mudhalvan  
- **Institution:** 1105 - Gojan School of Business and Technology  
- **Course:** Data Analytics in Process Industries  
- **Project Title:** Classification of Water Safety  

---

## 🧪 Problem Statement

Ensuring access to safe drinking water is a critical public health concern. Traditional testing methods can be time-consuming and inaccessible in many regions. This project leverages machine learning to classify water samples based on indicators like pH, conductivity, and contaminants (e.g., arsenic, lead, nitrates), offering a fast and scalable alternative.

---

## 🛠️ Tech Stack

- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn** (for EDA & data handling)  
- **Scikit-learn** (for ML modeling)  
- **SMOTE** (for imbalance handling)  
- **Streamlit** (for web application)  
- **Git** (version control)

---

## 🧰 Skills Used

- Data Preprocessing & Cleaning  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Model Building (Classification)  
- Hyperparameter Tuning (GridSearchCV)  
- Model Evaluation  
- Streamlit App Development  
- Deployment Readiness

---

## 📊 Dataset

- **Name:** `water-quality-data.csv`
- **Attributes Include:**  
  - `pH`, `Dissolved Oxygen`, `Conductivity`, `Turbidity`, `Arsenic`, `Lead`, `Nitrates`, etc.
  - `is_safe` (target variable: 1 = Safe, 0 = Unsafe)

---

## 📈 Project Pipeline

### 1. Data Cleaning
- Fix inconsistent formats (e.g., ammonia column).
- Handle missing or outlier values.

### 2. Exploratory Data Analysis (EDA)
- Visualize feature distributions.
- Analyze correlation heatmaps.
- Identify feature impact on water safety.

### 3. Imbalanced Data Handling
- Applied **SMOTE** to handle minority class.
- Compared results with and without rebalancing.

### 4. Feature Engineering
- Scaled numerical features.
- Removed redundant features.

### 5. Model Training
- Models Used:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- **Best Model:** Random Forest with tuned parameters.

### 6. Evaluation Metrics
- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC-AUC Curve**

### 7. Streamlit App
- Upload `.csv` file to predict water safety.
- Displays prediction results and plots.

---

## 🚀 Deployment

✅ Streamlit App ready for deployment.

To run locally:
```bash
streamlit run app.py
