# Kuzushiji-MNIST Classification Project

An end-to-end Machine Learning project to classify ancient Japanese characters (Kuzushiji) using Computer Vision techniques. This repository contains the Exploratory Data Analysis (EDA), feature engineering, and model optimization pipelines.

## 📌 Project Overview
The goal of this project was to replicate a Data Science workflow on the KMNIST dataset, a more challenging alternative to the classic MNIST digits. The analysis focuses on:
- **Exploratory Data Analysis (EDA):** Using heatmaps and Euclidean distance matrices to understand class variability.
- **Binary Classification:** Comparing feature selection strategies (Top N vs. Random) using KNN.
- **Multiclass Classification:** Optimizing Decision Trees using K-Fold Cross Validation.

## 🛠️ Technologies
- **Python**
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy.

## 📊 Key Findings & Results
1. **EDA:** We discovered that "average images" can be misleading. For example, Class 2 is structurally similar to Class 6, a confusion pattern later confirmed by the model's errors.
2. **KNN Optimization:** Achieved an **Accuracy of 0.932** in binary classification by optimizing *k* and the number of features (*N*).
3. **Decision Tree:** Through GridSearch and 5-Fold Cross Validation, the optimal hyperparameters found were `max_depth=10` and `min_samples_split=10`.

## 📂 Repository Structure
- `main.py`: Contains the full script with EDA, training, and evaluation logic.
- `informe.pdf`: Detailed academic report with graphs and theoretical justification.

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run the script:
   ```bash
   python main.py
