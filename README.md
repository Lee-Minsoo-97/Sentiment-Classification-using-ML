# Sentiment-Classification-using-ML
Sentiment Classification with Machine Learning Approaches


# 📊 Sentiment Analysis Using Machine Learning

This repository contains a comprehensive sentiment analysis project conducted by **Minsoo Lee**. The goal of this project is to classify customer reviews into **positive**, **neutral**, and **negative** sentiments using various text processing techniques and machine learning models.

## 🚀 Project Overview

- **Text Preprocessing:** Tokenization, POS Tagging, Lemmatization, Named Entity Recognition (NER), Parsing
- **Feature Extraction:** Bag of Words (BoW) using CountVectorizer and TF-IDF Vectorization
- **Sentiment Classification Models:**
  - Lexicon-Based Approach (VADER)
  - Naive Bayes Classifier
  - Support Vector Machines (SVM)
  - Logistic Regression
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📁 Files in This Repository

- `LA2_Lee_Minsoo.ipynb` - The complete Python notebook with code, outputs, and analysis.
- `Notebook_Output.html` - Rendered HTML file of the notebook.
- `README.md` - This file.

---

## 🛠️ How to Run This Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Open with Google Colab:**

   - Upload the notebook to Colab or run directly if cloned from GitHub.

3. **Install required packages:**

   ```python
   !pip install pandas numpy scikit-learn nltk spacy seaborn matplotlib
   !python -m spacy download en_core_web_sm
   ```

4. **Run all cells to see the outputs.**

---

## 📊 Results Summary

- **VADER (Lexicon-Based):** Accuracy: 86%
- **Naive Bayes:** Accuracy: 86%
- **SVM:** Accuracy: 89%
- **Logistic Regression (TF-IDF):** Accuracy: 85%

SVM performed the best with the highest accuracy, while VADER showed robust results in lexicon-based analysis.

---

## 📌 Key Observations

- Lexicon-based approaches like VADER are fast but may lack deep contextual understanding.
- Machine learning models, especially SVM, perform better with sufficient training data.
- Logistic Regression with TF-IDF captures more nuanced sentiment features.

---

## 👤 Author

**Minsoo Lee**\
Business Analytics, W. P. Carey School of Business\

