# University Student Dropout Prediction: Early Warning System using Machine Learning

## Project Overview
This project develops a predictive classification system designed to identify university students at high risk of dropping out. By anticipating student attrition early on, educational institutions can intervene proactively through tutoring or financial aid, improving retention rates and mitigating the economic impact associated with student loss.

The analysis distinguishes itself by integrating not only the student's academic history but also demographic factors and macroeconomic variables (such as inflation, GDP, and unemployment rates), providing a novel contextual layer compared to traditional models.

## Dataset
The data is sourced from the **UCI Machine Learning Repository**: [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). 
The dataset contains 4,424 instances and 36 features, originating from official academic records of the Instituto Politécnico de Portalegre (Portugal).

The target variable classifies the student's final status into three categories:
* Dropout
* Enrolled
* Graduate

## Methodology and Project Architecture

The project follows a comprehensive Data Science and Engineering lifecycle, structured into the following phases:

1. **Data Engineering and Preprocessing:**
   * Null value imputation and outlier filtering.
   * Normality and statistical distribution assessment.
   * Feature transformation: Standardization of continuous variables (`StandardScaler`) and categorical encoding (`One-Hot Encoding`).

2. **Exploratory Data Analysis (EDA) and Feature Selection:**
   * Univariate and bivariate descriptive statistics.
   * Correlation matrices to identify multicollinearity and reduce dimensionality.
   * Visual analysis of the socioeconomic context impact (Novelty Factor).

3. **Predictive Modeling:**
   * **Baseline Model:** Logistic Regression (used as a benchmark due to its high interpretability).
   * **Complex Models:** K-Nearest Neighbors (KNN), Support Vector Machines (SVM with non-linear kernel), Decision Trees, and Artificial Neural Networks.

4. **Business Evaluation and Cost Analysis:**
   * Prioritization of the **Recall** metric for the "Dropout" class to minimize False Negatives (students who drop out without being flagged by the system).
   * Comprehensive model comparison across five dimensions: Accuracy, Recall, Computational/Energy Cost, Novelty, and Interpretability.

## Repository Structure

```text
├── data/
│   ├── raw/                      # Original, unprocessed dataset and processing notebooks
│   └── processed/                # Transformed dataset (.csv) ready for training
├── model_scripts/                # Individual scripts for each model
├── docs/                         # Additional documentation and final report
├── .gitignore                    # Files and directories excluded from version control
├── requirements.txt              # Project dependencies
└── README.md                     # Project description

## Technologies Used
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Modeling, Preprocessing, Metrics)
* **Version Control:** Git and GitHub

## Reproducibility
To run this project in a local environment, follow these instructions:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/repo-name.git](https://github.com/your-username/repo-name.git)
   cd repo-name
   ```

2. **Create and activate a virtual environment:**
   * On Windows:
     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   * On macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
## Conclusion and Application
This analysis demonstrates the viability of using Machine Learning algorithms not just as technical tools, but as strategic instruments for institutional decision-making. It establishes a clear trade-off between the high predictive capacity of algorithms like Neural Networks and the interpretative transparency of Decision Trees, with the latter being highly valuable for academic tutors to understand the reasoning behind each prediction.