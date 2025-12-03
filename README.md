# 🧠 Customer Category Classifier
### Machine Learning & AI Project (CSE422)

This project develops a machine learning–based system to **predict whether a customer has ever been married** using demographic, professional, and behavioral features. Multiple supervised and unsupervised ML techniques were applied, including **KNN, Decision Tree, Neural Network**, and **K-Means Clustering**.

---

## 📌 Project Overview

* **Problem Type:** Binary Classification
* **Target Variable:** `Ever_Married` (Yes = 1, No = 0)
* **Dataset Size:** 24,210 customer records
* **Features:** Age, Gender, Profession, Spending Score, Work Experience, Family Size
* **Goal:** Build predictive models and analyze customer segments for marketing and recommendation use cases.

---

## 📊 Dataset Description

* Contains **both numerical and categorical** features.
* Categorical features were encoded using **One-Hot Encoding**.
* Dataset was slightly imbalanced:
  * Married: 60%
  * Not Married: 40%

### ✔ Preprocessing Steps
* Missing values handled (mean for numeric, mode for categorical)
* One-Hot Encoding for categorical variables
* Standard scaling for numerical features (important for KNN)
* Stratified 70/30 train-test split

---

## 🧪 Models Implemented

### **1️⃣ K-Nearest Neighbors (KNN)**
* **k = 5**
* **Accuracy:** 84%
* Performs reasonably but sensitive to high-dimensional encoded features.

### **2️⃣ Decision Tree Classifier**
* **Max Depth = 6**, Criterion = Gini
* **Accuracy:** 87%
* Easy to interpret and balanced performance.

### **3️⃣ Neural Network (Best Model)**
**Architecture:**
* Input → Dense(64, ReLU, Dropout 0.3)
* Dense(32, ReLU)
* Output → Sigmoid

**Training:**
* Optimizer: Adam
* Loss: Binary Crossentropy
* Epochs: 50
* Batch Size: 32

**Performance:**
* **Accuracy: 88.1%**
* **AUC: 0.95**
* Best at capturing non-linear relationships.

### **4️⃣ K-Means Clustering (Unsupervised)**
* Best K = **4** (via Elbow Method)
* Visualized using PCA
* Revealed four distinct customer segments

---

## 📈 Model Evaluation Summary

| Model | Accuracy | Strength |
| :--- | :--- | :--- |
| **Neural Network** | **88.1%** | Best overall performance |
| **Decision Tree** | 87% | Interpretable and competitive |
| **KNN** | 84% | Simple but weaker on minority class |

Additional visual analyses included:
* Confusion Matrices
* Precision / Recall / F1-score
* ROC & AUC curves

---

## 📌 Key Findings
* Strong relationship between **Age**, **Profession**, and marital status.
* Spending score patterns differ across clusters and marital groups.
* Neural Network outperforms classical ML models.
* Clustering reveals hidden customer segments beyond marital status.

---

## 🚀 Future Improvements
* Hyperparameter tuning (GridSearch, RandomSearch)
* Add ensemble models (RandomForest, XGBoost)
* Apply SMOTE for class imbalance
* Explore advanced deep learning architectures (LSTM, Wide & Deep models)

---

## 📂 Project Structure (Suggested)

```text
Customer-Category-Classifier/
│
├── data/
│   └── Customer_Category_Classifier_Dataset.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── models/
│   └── saved_models/
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── clustering.py
│
├── README.md
└── requirements.txt
