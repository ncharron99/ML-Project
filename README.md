# 🐧 Penguin Species Classifier

> A machine learning class project to classify penguin species using physical and categorical traits from the Palmer Penguins dataset.

---

## 📌 Overview

We explored and modeled the classification of three penguin species:

* **Adelie**
* **Chinstrap**
* **Gentoo**

Using supervised learning algorithms, we performed:

* ✅ Data cleaning
* 📊 Exploratory data analysis
* 🔍 Feature selection
* 🤖 Model training and evaluation

### Models Used

* K-Nearest Neighbors (KNN)
* Random Forest Classifier
* Support Vector Machine (SVM)

---

## 🧠 Group Contributions

| Member        | Contributions                                                           |
| ------------- | ----------------------------------------------------------------------- |
| **Alan**      | Data cleaning, third figure, Random Forest model, discussion, revisions |
| **Clarence**  | First figure, KNN model, tables, feature selection, revisions           |
| **Nathaniel** | Second figure, SVM model, feature selection, revisions                  |

---

## 📂 Data Cleaning

**Steps Taken:**

* Removed the `Comments` column
* Dropped rows with missing values
* Filtered invalid entries in the `Sex` column

```python
def cleaning_data(data):
    data = data.drop(['Comments'], axis=1)
    data = data.dropna()
    if (data['Sex'] == '.').any():
        data = data.drop(data[data['Sex'] == '.'].index)
    return data
```

---

## 📊 Exploratory Data Analysis

We examined:

* **Boxplots** of quantitative features
* **Histograms** of body mass
* **Scatter plots** of Culmen Length vs. Depth

### Key Insights

* **Culmen Length** and **Depth** were strong predictors.
* **Body Mass** showed significant overlap between species.
* **Island** emerged as a highly predictive categorical feature.

---

## 🔍 Feature Selection

We used 4-fold cross-validation on all 3-feature combinations:

> Best feature set: `Culmen Length (mm)`, `Culmen Depth (mm)`, `Island`

---

## 🤖 Model Performance

### 🔎 K-Nearest Neighbors (KNN)

* **Best k:** 9
* **Accuracy:** 98.46%
* One Chinstrap penguin misclassified

### 🌳 Random Forest

* **Best depth:** 5
* **Accuracy:** 100%
* No misclassifications

### ⚖️ Support Vector Machine (SVM)

* **Best γ (gamma):** 0.006
* **Accuracy:** 95.39%
* Minor confusion between Chinstrap and Gentoo

---

## 📈 Visualizations

We created:

* � Boxplots & Histograms
* 🔺 Scatter plots
* 🎨 Decision region graphs
* 📊 Confusion matrices

These helped identify which features were most separable and how models behaved across islands.

---

## 📌 Key Discussion Points

* All models exceeded 95% accuracy
* Random Forest was the best performer
* Body Mass & Sex were not useful predictors
* **Suggested Improvement:** Use an "island-first" model-fitting strategy to refine decision regions

---

## 📁 Tech Stack

* Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* Environment: Jupyter Notebook
* Dataset: Palmer Penguins

---

## 🎯 Example Usage with KNN

<img width="557" alt="knn_confusion_matrix" src="https://github.com/user-attachments/assets/6a6c0c72-49ac-40ae-8f9b-0705190d42d0" />
<img width="1189" alt="KNN" src="https://github.com/user-attachments/assets/c3750d9d-f352-4f0a-bb95-3d67941b72a0" />
