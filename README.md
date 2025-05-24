# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*Company*: Codetech it solutions

*Name* : Naveen Kumar Ravi

*Intern Id* : CT04DN1399

*Domain* : Python Programming

*Duration* : 4 Weeks

*Mentor* : Neela Santhosh Kumar

# 📦 Machine Learning Project - Spam Email Detection using Scikit-learn

This project is part of the **Machine Learning Model Implementation Internship** at **CodTech**. The goal of the project is to create a predictive model using Scikit-learn to classify emails as **Spam** or **Ham (Not Spam)** using machine learning techniques.

---

## 🎯 Objective

To build a predictive classification model using Scikit-learn that can accurately identify whether a given message is **spam** or **ham** (non-spam). This project involves:

- Data preprocessing
- Feature extraction
- Model training
- Evaluation
- Saving the model for deployment

---

## 🧠 Tools & Libraries Used

- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib & Seaborn (for visualization)
- NLTK (for text preprocessing)
- Jupyter Notebook

---

## 📁 Dataset

**Dataset Name**: SMS Spam Collection  
**Source**: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) or [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
**Format**: CSV/TSV  
**Attributes**:
- `label`: Indicates whether the message is `ham` or `spam`
- `message`: The text of the SMS message

---

## 🔍 Project Workflow

### 1. Data Collection
The dataset is loaded from a remote or local source and stored using Pandas DataFrame.

### 2. Data Preprocessing
- Convert text to lowercase
- Remove punctuation
- Remove stopwords (using NLTK)
- Apply stemming

### 3. Feature Extraction
TF-IDF Vectorizer is used to convert text into numerical feature vectors.

### 4. Model Building
A **Multinomial Naive Bayes** model is trained on the processed dataset.

### 5. Model Evaluation
We evaluate the model using:
- Accuracy
- Classification Report (Precision, Recall, F1 Score)
- Confusion Matrix (with a heatmap)

### 6. Model Saving
The model and vectorizer are saved using Python's `pickle` module for future deployment.

---

## ⚙️ How to Run This Project

### 🔧 Prerequisites
Install required packages:
```bash
pip install pandas scikit-learn matplotlib seaborn nltk
