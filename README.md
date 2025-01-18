# Wine Quality Prediction

This project focuses on building and evaluating machine learning models to predict wine quality based on its physicochemical properties. Using data from the UCI Machine Learning Repository, the project explores feature engineering, data preprocessing, model selection, and hyperparameter tuning to achieve the best classification performance.

---

## **Project Objectives**
1. **Analyze the Wine Dataset**:
   - Understand the structure, distribution, and key statistics of the dataset.
   - Visualize the relationships between features and the target variable.

2. **Feature Engineering**:
   - Create new features to enhance the predictive power of the dataset.
   - Engineer features such as polynomial transformations, interaction terms, and ratios.

3. **Data Balancing**:
   - Address the class imbalance issue using techniques such as **SMOTE** to ensure fair model evaluation.

4. **Feature Selection**:
   - Apply multiple feature selection techniques (e.g., RFE, Lasso Regression, Random Forest) to identify the most important features.

5. **Model Building**:
   - Train and evaluate different machine learning classifiers:
     - **Random Forest**
     - **K-Nearest Neighbors (KNN)**
     - **Logistic Regression**
   - Tune hyperparameters using **Grid Search** to optimize model performance.

6. **Evaluate Performance**:
   - Measure model performance using metrics like accuracy, precision, recall, and F1-score.
   - Analyze the confusion matrix and classification report to assess class-level performance.

---

## **Dataset**
- **Name**: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Source**: UCI Machine Learning Repository
- **Description**:
  - Contains 6,497 samples of wine.
  - Features include physicochemical properties like acidity, sulfur dioxide levels, and alcohol content.
  - Target variable: Quality (scored from 3 to 9).

---

## **Key Steps**
1. **Data Preprocessing**:
   - Normalize continuous features using **MinMaxScaler**.
   - Visualize the feature distributions and relationships with the target variable.
   - Generate a correlation heatmap to identify highly correlated features.

2. **Feature Engineering**:
   - New features:
     - `alcohol_to_volatile_acidity_ratio`: Ratio of alcohol to volatile acidity.
     - `free_minus_total_sulfur_ratio`: Difference between free and total sulfur dioxide levels.
     - `total_acidity_to_alcohol_ratio`: Combined acidity relative to alcohol content.
   - Polynomial and interaction features:
     - `pH_squared`
     - `density_sugar_interaction`
     - `log_total_sulfur_dioxide`

3. **Feature Selection**:
   - Recursive Feature Elimination (**RFE**) with Random Forest.
   - Lasso Regression with cross-validation.
   - Feature importance from Random Forest.

4. **Data Balancing**:
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.

5. **Model Building**:
   - Built and evaluated models using features selected by RFE and Random Forest.
   - Tuned hyperparameters with **GridSearchCV** to optimize performance.
   - Evaluated models with a **classification report**, accuracy, and confusion matrix.

---

## **Key Findings**
1. **Feature Importance**:
   - Features like `alcohol`, `volatile_acidity`, and `density` were consistently selected as the most important across feature selection methods.
   
2. **Best Performing Model**:
   - **Random Forest** with features selected by Random Forest performed the best:
     - Accuracy: ~70%
     - Weighted F1-Score: ~0.69
   - Hyperparameter tuning using Grid Search improved performance slightly.

3. **Data Balancing**:
   - SMOTE improved recall for underrepresented classes (e.g., quality scores 3 and 4), but overall accuracy remained stable.

---

## **Project Structure**
wine-quality-prediction/ │ ├── data/ │ ├── wine_dataset.csv # Original dataset │ ├── preprocessed_data.csv # Preprocessed dataset │
├── notebooks/ │ ├── data_analysis.ipynb # EDA and visualization │ ├── feature_engineering.ipynb # Feature engineering and selection │ 
├── model_training.ipynb # Model training and evaluation │ ├── models/ │ ├── best_model.pkl # Saved Random Forest model │ ├── results/ │ 
├── classification_report.txt # Model performance summary │ ├── visualizations/ # Plots and graphs │
├── README.md # Project documentation ├── requirements.txt # Python dependencies └── main.py # Main script for running the project


---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/wine-quality-prediction.git
   cd wine-quality-prediction
2. Install dependencies:
pip install -r requirements.txt
3. Run the project:
   Jupyter Notebooks: Use notebooks/ to follow step-by-step analysis.
   Main Script: Execute main.py for a streamlined workflow.
