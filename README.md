
# 📊 Customer Churn Prediction using Machine Learning

This project aims to predict customer churn in the telecommunications industry using supervised machine learning models. By analyzing customer attributes and subscription data, it helps businesses identify high-risk customers likely to churn and take preventive actions.

---

## 📁 Dataset

- **Source**: IBM Sample Data Set – *Telco Customer Churn*
- **Rows**: 7,043
- **Columns**: 21
- **Target variable**: `Churn` (Yes/No)
- **Key Features**:
  - `gender`, `SeniorCitizen`, `Partner`, `tenure`, `InternetService`, `Contract`, `PaymentMethod`, etc.

---

## 🚀 Project Highlights

- Cleaned and preprocessed raw data (handled missing values, converted `TotalCharges` from object to numeric).
- Used **Label Encoding** and **One-Hot Encoding** for categorical variables.
- Built predictive models using **Logistic Regression** and **Random Forest Classifier**.
- Achieved ~85% accuracy on the test set with Random Forest.
- Saved trained model and encoders using `joblib` for future inference.

---

## 🧰 Technologies Used

- Python, Pandas, NumPy, Matplotlib
- Scikit-learn (modeling, preprocessing, evaluation)
- Jupyter Notebook
- Joblib (for model persistence)

---

## 📦 Installation & Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
   ```

---

## ⚙️ How to Use

- To test the model:
  - Load `encoders.pkl` and `dustomer_churn_model.pkl`
  - Prepare input in the same format as training features
  - Use `model.predict()` to generate churn predictions

---

## 📈 Evaluation Metrics

- **Accuracy**, **Precision**, **Recall**, **F1 Score**
- Confusion matrix and classification report used for detailed evaluation

---

## 📎 Files in this Repository

| File | Description |
|------|-------------|
| `Customer_Churn_Prediction_using_ML.ipynb` | Main notebook with full workflow |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |
| `dustomer_churn_model.pkl` | Trained Random Forest model |
| `encoders.pkl` | Encoded mapping for categorical features |

---

## 🙌 Acknowledgments

- Dataset by IBM Watson Studio
- Inspired by real-world telecom churn use cases
