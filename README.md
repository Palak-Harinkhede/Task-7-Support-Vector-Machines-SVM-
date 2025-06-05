# 🧠 Task 7: Support Vector Machines (SVM) - Breast Cancer Classification

## 📌 Objective

Use Support Vector Machines (SVMs) for binary classification using both **Linear** and **RBF** kernels, along with visualization and hyperparameter tuning.

---

## 📂 Dataset

**Name:** Breast Cancer Dataset  
**Source:** [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)  
**Description:** Contains diagnostic data for breast cancer tumors labeled as malignant (M) or benign (B), with 30 numerical features.

---

## 🔧 Tools Used

- Python 🐍
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## 🚀 Steps Followed

### 1. Load and Explore the Dataset
- Loaded the dataset using Pandas
- Checked for missing values
- Inspected class balance

### 2. Preprocessing
- Converted target variable `diagnosis` to binary (`M` → 1, `B` → 0)
- Removed unnecessary columns like `id`
- Applied StandardScaler for normalization
- Split dataset into train and test sets

### 3. Train SVM with Linear Kernel
- Used `SVC(kernel='linear')`
- Evaluated with accuracy, precision, recall, F1-score

### 4. Train SVM with RBF Kernel
- Used `SVC(kernel='rbf')`
- Compared performance with linear kernel

### 5. Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation
- Tuned parameters: `C`, `gamma`

### 6. 2D Visualization (with PCA)
- Reduced features to 2D using PCA
- Plotted decision boundary using `matplotlib`

---

## 📈 Results

| Kernel | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Linear | ~96.5%   | High      | High   | High     |
| RBF    | ~98.2%   | Higher    | Higher | Higher   |

> 🔍 RBF kernel outperformed linear kernel due to its non-linear transformation ability.

---

## 📊 Visualization

<img src="decision_boundary.png" alt="SVM Decision Boundary" width="600"/>

- 2D visualization of decision boundary after PCA transformation.

---

## 🧠 Interview Q&A

| Question | Answer |
|---------|--------|
| **What is a support vector?** | Data points closest to the decision boundary that influence it. |
| **What does the C parameter do?** | Controls trade-off between smooth margin and classification accuracy. |
| **What are kernels in SVM?** | Functions that project data into higher-dimensional space for better separation. |
| **Linear vs RBF kernel?** | Linear is used when data is linearly separable; RBF is for non-linear data. |
| **Advantages of SVM?** | Effective in high-dimensional space, memory efficient, robust to overfitting. |
| **Can SVM be used for regression?** | Yes, via Support Vector Regression (SVR). |
| **What if data isn’t linearly separable?** | Use kernel trick like RBF to separate it in higher dimensions. |
| **How is overfitting handled?** | By tuning `C` and `gamma`, and using cross-validation. |

---


