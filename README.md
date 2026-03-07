# 🌍 SDG Health Analytics — Life Expectancy Modeling

SDG-Health Analytics: Life Expectancy Modeling is a comprehensive data science project focused on predicting life expectancy in developing countries using sanitation access, infant mortality, and related health indicators.

Developed as part of the **IIMSTC Internship**, this study aligns with **United Nations Sustainable Development Goal 3 (SDG 3): Good Health and Well-Being**.

The core objective of this project is to move beyond simple correlation analysis and build a predictive framework that quantifies how infrastructure (sanitation) and early-life health outcomes (mortality) influence population longevity.

---

## 🎯 Project Objectives

- Analyze the relationship between sanitation access, infant mortality, and life expectancy.
- Perform structured data preprocessing and feature engineering.
- Build multiple regression-based predictive models.
- Compare model performances using statistical evaluation metrics.
- Support data-driven public health insights aligned with SDG 3.

---

## 📊 Dataset Overview

The Life Expectancy dataset (sourced from Kaggle) includes:

- Life Expectancy (Target Variable)
- Infant Mortality Rate
- Sanitation Access
- GDP & Economic Indicators
- Immunization Rates
- Health Expenditure
- Education & Demographic Factors

The dataset was filtered to focus on developing countries and extensively cleaned before model implementation.

---
# 👥 Team Contributions

---

## 👨‍💻 Vaastav L Sanghvi  (1DT22CD051)
### Role: Project Lead – Exploratory Data Analysis & Polynomial Regression

### Responsibilities:
- Led the analytical direction of the project.
- Conducted comprehensive Exploratory Data Analysis (EDA) to understand distributions, correlations, and feature behavior.
- Performed correlation heatmap analysis and statistical relationship validation.
- Applied log transformations to address skewness in health indicators.
- Identified and handled outliers for improved model stability.
- Developed and evaluated the Polynomial Regression model to capture non-linear relationships.
- Coordinated model comparison across Linear, Multiple, Ridge, and Polynomial Regression approaches.

### Key Contributions:
- Identified strong predictors of Life Expectancy through structured EDA.
- Demonstrated non-linear impact of sanitation and mortality indicators.
- Strengthened interpretability through residual and diagnostic visualizations.
- Contributed to overall performance benchmarking and insight generation.

---

## 👩‍💻 Sruthi K S (4CB22CS136)  
### Role: Data Collection & Preprocessing

### Responsibilities:
- Collected dataset from Kaggle.
- Cleaned missing values and removed duplicates.
- Filtered developing countries.
- Applied outlier removal and normalization.
- Prepared structured dataset for modeling.

### Outcome:
- Generated `life_expectancy_cleaned.csv`.
- Improved data consistency and quality.

---
**Avinash**
Primary Responsibility: Performed Exploratory Data Analysis (EDA) to understand data patterns and support the modeling team.

🔍 Key Tasks:

Conducted statistical analysis (mean, median, standard deviation).

Identified skewed distributions and outliers.

Performed correlation analysis and detected multicollinearity.

Created visualizations including heatmaps, distribution plots, and trend analysis graphs.

📦 Deliverables:

Complete EDA report.

Insight summary with feature importance and preprocessing recommendations for the modeling team.

## 👨‍💻 Vrushank Skanda B (1JT22AI059)  
### Role: Feature Engineering

### Responsibilities:
- Applied Median Imputation for robustness.
- Engineered composite indices and interaction features.
- Used MinMaxScaler before index generation.
- Ensured leakage-free preprocessing via proper Train-Test split.
- Standardized temporal features.

### Outcome:
- Created high-correlation engineered features.
- Exported:
  - `life_expectancy_train_master.csv`
  - `life_expectancy_test_master.csv`

---

## 👨‍💻 Rafa Rahmath (1HK22CS114)  
### Role: Multiple Linear & Ridge Regression Modeling

### Responsibilities:
- Implemented Multiple Linear Regression.
- Conducted VIF analysis for multicollinearity.
- Built ML pipeline integrating scaling and imputation.
- Applied Ridge Regression for regularization.

### Outcome:
- Ridge Regression achieved Test R² = 0.8041.
- Ensured strong generalization and reduced overfitting.

---

## 👩‍💻 Sadiya Kulsum (1HK22CS124)  
### Role: Linear Regression Modeling

### Responsibilities:
- Built baseline Linear Regression model.
- Performed multicollinearity checks using VIF.
- Integrated preprocessing pipeline.
- Conducted residual diagnostics.

### Outcome:
- Achieved Test R² = 0.85.
- Provided interpretable baseline model.

---

## 👨‍💻 Saayanth M (1ST22AI042)  
### Role: End-to-End Data Preparation & EDA Support

### Responsibilities:
- Performed encoding and feature scaling.
- Conducted exploratory analysis of economic and health factors.
- Assisted in regression implementation.

### Outcome:
- Identified key influencing variables.
- Improved model performance via preprocessing refinement.

---

## 👨‍💻 Ravi (1BO22EC048)  
### Role: Model Evaluation & Reporting

### Responsibilities:
- Implemented Simple, Multiple, and Polynomial Regression comparisons.
- Evaluated models using R² and error metrics.
- Prepared analytical report.

### Outcome:
- Confirmed sanitation and infant mortality as major determinants.
- Supported SDG-aligned public health insights.

# 📈 Key Insights

- Infant Mortality Rate shows strong negative correlation with Life Expectancy.
- Sanitation Access significantly improves longevity.
- Regularization (Ridge) improves generalization.
- Proper preprocessing enhances predictive accuracy.
- Feature engineering significantly strengthens model performance.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 🏗 Project Structure
SDG-Health-Analytics-Life-Expectancy-Modeling/
│
├── data/
├── notebooks/
├── reports_and_visuals/
├── life_expectancy_cleaned.csv
├── life_expectancy_train_master.csv
├── life_expectancy_test_master.csv
└── README.md

---
🌎 SDG Impact

This project contributes to SDG 3: Good Health and Well-Being by:

Identifying major health determinants.

Supporting evidence-based policymaking.

Demonstrating the role of machine learning in public health analytics.

👨‍💻 Developed As Part Of

IIMSTC Internship Program

📄 License

This project is for academic and research purposes.



