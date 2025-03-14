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

3️⃣ Model Selection & Training
    Tested several classification models to find the best one:

Model	Accuracy	F1-Score	Precision	Recall
Logistic Regression	0.9720	0.9636	0.9464	0.9815
Decision Tree	0.9231	0.8972	0.9057	0.8889
Random Forest	0.9790	0.9725	0.9636	0.9815
K-Nearest Neighbors (KNN)	0.9650	0.9550	0.9298	0.9815
Support Vector Classifier (SVC)	0.9720	0.9636	0.9464	0.9815
Gradient Boosting	0.9580	0.9455	0.9286	0.9630
📌 Best Model: Random Forest achieved the highest accuracy (97.9%) and balanced performance.

4️⃣ Hyperparameter Tuning
We fine-tuned Random Forest using GridSearchCV to optimize hyperparameters:
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
jupyter notebook BREAST_cancer_-.ipynb

📎 Author
👤 Ajmal Bilal K 
