🐧 Penguin Species Classifier

This project applies machine learning techniques to classify penguin species using the Palmer Penguins dataset. It includes exploratory data analysis, feature selection, model evaluation, and visualization. Our goal was to determine which models and features most accurately predict the species of a penguin based on physical characteristics and categorical traits.

📌 Project Overview

We explored and modeled the classification of three penguin species — Adelie, Chinstrap, and Gentoo — using various supervised learning algorithms. We performed data cleaning, visualization, and feature selection before training and comparing the performance of the following models:

- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Machine (SVM)

🧠 Group Contributions

Member

Contributions

Alan

Data cleaning, third figure, third model (Random Forest), discussion, revisions

Clarence

First figure, first model (KNN), table creation, feature selection with Nathaniel, revisions

Nathaniel

Second figure, second model (SVM), feature selection with Clarence, revisions

📂 Data Cleaning

We removed:

The Comments column.

Rows with missing values.

Rows with invalid entries (e.g., Sex column containing ".").

def cleaning_data(data):
    data = data.drop(['Comments'], axis=1)
    data = data.dropna()
    if (data['Sex'] == '.').any():
        data = data.drop(data[data['Sex'] == '.'].index)
    return data

📊 Exploratory Data Analysis

We visualized:

Distributions of quantitative features (culmen length/depth, flipper length, body mass).

Histograms of body mass per species.

A scatter plot of culmen length vs. culmen depth, which revealed distinct clusters.

Key insights:

Culmen Length and Culmen Depth were the most discriminative features.

Body Mass had significant overlap between species and was less useful for prediction.

Island feature appeared promising due to distinct species-to-island mappings.

🔍 Feature Selection

We evaluated combinations of quantitative and categorical features using 4-fold cross-validation on a Logistic Regression model. The top-performing combination:

Culmen Length (mm)

Culmen Depth (mm)

Island

🤖 Models and Evaluation

1. K-Nearest Neighbors (KNN)

Best k: 9

Test Accuracy: 98.46%

Confusion matrix: 1 Chinstrap misclassified.

2. Random Forest Classifier

Best max depth: 5

Test Accuracy: 100%

Clean decision boundaries, no misclassifications.

3. Support Vector Machine (SVM)

Best γ (gamma): 0.006

Test Accuracy: 95.39%

A few Gentoo and Chinstrap penguins misclassified.

📈 Visualizations

We created:

Boxplots and histograms to examine feature distributions.

Scatter plots to identify separability of features.

Decision region plots for each classifier by island.

Confusion matrices to evaluate classification performance.

📌 Key Discussion Points

All models performed above 95% accuracy, with Random Forest achieving perfect accuracy.

Features chosen through cross-validation proved highly effective.

Body Mass and Sex were not informative for classification.

Suggested Improvement: An "iterative prediction" strategy where models are fit in two stages — first by Island, then by culmen measurements — could enhance classification performance by reducing noise in decision regions.

📁 Technologies Used

Python (Pandas, NumPy, scikit-learn, Matplotlib, Seaborn)

Jupyter Notebooks

Palmer Penguins dataset

Example Usage: 
<img width="557" alt="knn_confusion_matrix" src="https://github.com/user-attachments/assets/6a6c0c72-49ac-40ae-8f9b-0705190d42d0" />
<img width="1189" alt="KNN" src="https://github.com/user-attachments/assets/c3750d9d-f352-4f0a-bb95-3d67941b72a0" />
