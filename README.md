# 💼 Salary Prediction using Random Forest

This project predicts whether an individual's salary is >50K or <=50K based on census data using the **Random Forest Classifier**. The dataset used is the **Adult Income Dataset (Census Income)** from the UCI Machine Learning Repository.

## 📂 Dataset Description

- **Source**: UCI Machine Learning Repository / Kaggle
- **Features**:
  - Age
  - Workclass
  - Education
  - Marital Status
  - Occupation
  - Relationship
  - Race
  - Sex
  - Capital Gain
  - Capital Loss
  - Hours per week
  - Native country
- **Target**: Salary (<=50K or >50K)

## 🧪 Objective

To build a machine learning model that predicts whether a person earns more than \$50K per year using socio-economic features, and evaluate the model's performance using standard metrics.

---

## 🧹 Data Preprocessing

✔️ Handled missing values  
✔️ Label encoded categorical features  
✔️ Scaled numerical features using StandardScaler  
✔️ Train-test split (80-20)

---

## 📊 Exploratory Data Analysis (EDA)

- Age distribution
- Salary distribution across genders
- Relationship between education and income
- Hours worked vs. salary
- Feature correlation heatmap

---

## 🤖 Model Used

- ✅ **Random Forest Classifier** (with default and tuned hyperparameters)

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

---

## 🛠️ Tools & Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook / Google Colab

---

## 🧠 Results

- Achieved a high classification accuracy
- Random Forest outperformed simple baseline models
- Good balance of precision and recall

---

