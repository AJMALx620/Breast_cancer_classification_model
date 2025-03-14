Breast Cancer Classification Using Machine Learning

📌 Project Overview

This project focuses on classifying breast cancer tumors as malignant or benign using multiple machine learning models. The dataset used is well-structured and contains key medical features that help predict cancer type.

📂 Dataset Information

Source:
This dataset was collected for breast cancer classification. It has been cleaned and preprocessed for model training.

Target Variable: Diagnosis (Malignant/Benign)

Features: Various cell nucleus properties such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, etc.

Goal: Develop a reliable model to assist in early cancer detection.


🚀 Machine Learning Workflow


1️⃣ Data Preprocessing

✔ Handled missing values and cleaned the dataset.

✔ Converted categorical labels into numerical form.

✔ Standardized features to improve model performance.


2️⃣ Train-Test Split

✔ The dataset was split into training (80%) and testing (20%) to ensure fair evaluation.

📌 Best Model: Random Forest achieved the highest accuracy (97.9%) and balanced performance.


4️⃣ Hyperparameter Tuning

Fine-tuned Random Forest using GridSearchCV to optimize hyperparameters:

✔ n_estimators: 50, 100, 200

✔ max_depth: 10, 20, None

✔ min_samples_split: 2, 5, 10

✔ min_samples_leaf: 1, 2, 4


📌 Fine-Tuned Random Forest Performance:

Accuracy: 97.2%
F1-Score: 96.3%


5️⃣ Model Evaluation

✔ Evaluated models using accuracy, precision, recall, and F1-score.

✔ Avoided overfitting by comparing results on training vs. test sets.


# Conclusion

✅ The Random Forest model performed best after hyperparameter tuning.

✅ This model can assist doctors in early breast cancer detection with high accuracy.

✅ Future work can explore deep learning techniques for further improvement.


📜 How to Run the Notebook


Install required dependencies:

#pip install numpy pandas scikit-learn seaborn matplotlib


Open the Jupyter Notebook and run all cells:

#jupyter notebook BREAST_cancer_-.ipynb

📎 Author

👤 Ajmal Bilal K 
