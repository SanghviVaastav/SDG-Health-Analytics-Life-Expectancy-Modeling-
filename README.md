# SDG-Health-Analytics-Life-Expectancy-Modeling-
SDG-Health Analytics: Life Expectancy Modeling is a data science project that predicts life expectancy in developing countries using sanitation access and infant mortality data. The project applies data analysis and regression models to identify key health factors and support insights aligned with SDG 3 – Good Health and Well-Being.

This project predicts Life Expectancy in developing countries by analyzing the impact of sanitation access and infant mortality rates. Developed as part of the IIMSTC Internship, the study aligns with UN Sustainable Development Goal 3 (SDG 3) to promote Good Health and Well-Being.

The core objective is to move beyond simple correlation and build a predictive framework that quantifies how infrastructure (sanitation) and early-life health outcomes (mortality) dictate population longevity

Rafa Rahmath (1HK22CS114) - Multi Linear Regression Model

Task Explanation :
The primary objective of this project was to implement a predictive model using multi-linear regression to analyze a dataset and determine how various independent variables influence a target numerical output. The process involved extensive data preprocessing to handle potential statistical issues, such as multicollinearity, which was initially flagged by runtime warnings during the calculation of Variance Inflation Factors (VIF). To address these complexities and prevent overfitting, the script utilized a sophisticated machine learning pipeline that integrated data imputation and scaling with multiple regression techniques, specifically focusing on the performance of Ridge Regression.

Outcome :
The analysis successfully identified the Ridge Regression model as the most effective approach for this specific dataset, outperforming other tested methods. This model achieved a notable level of accuracy, yielding a Test R2 score of 0.8041, which suggests that approximately 80.4% of the variance in the target variable can be explained by the included features. To ensure the reliability of these results, the project concluded with a series of diagnostic visualizations—including "Actual vs. Predicted" and residual plots—which confirmed that the model maintained a strong fit and satisfied the underlying assumptions of linear regression.

Sruthi K S (4CB22CS136)

Task Explanation
Collected the Life Expectancy dataset from Kaggle and performed complete data cleaning and preprocessing.
Handled missing values, removed duplicate records, filtered developing countries, and selected important features.
Applied outlier removal, log transformation, encoding, and normalization to prepare the dataset for model development.

Outcome of Work
Generated a fully cleaned and structured dataset ready for machine learning analysis.
Improved data quality by eliminating inconsistencies and scaling numerical features properly.
Saved the final processed dataset as life_expectancy_cleaned.csv for accurate model training and evaluation.

Saayanth M  (1ST22AI042)

Task Explanation:
Performed data preprocessing on the Life Expectancy dataset including handling missing values, encoding categorical variables, and feature scaling.
Conducted exploratory data analysis (EDA) to understand relationships between health, economic, and demographic factors affecting life expectancy.
Built and evaluated regression models to predict life expectancy based on selected features.

Task Outcomes:
Identified key factors such as income, immunization, and healthcare indicators significantly influencing life expectancy.
Successfully developed a predictive model with improved accuracy after proper preprocessing and feature engineering.
Gained hands-on experience in end-to-end data analytics workflow including data cleaning, visualization, modeling, and evaluation.

Vrushank Skanda B (1JT22AI059)

Task Explanation :
Robust Missing Value Handling: Identified null values with .isnull().sum() and applied Median Imputation. This is mathematically superior to mean imputation for this dataset as it prevents extreme GDP and Mortality outliers from biasing the data.Competition-Level Feature Engineering: Developed advanced predictors including Ratios (Health Expenditure vs. Mortality), Log Transformations (to handle skewed GDP), and Composite Indices. Components were normalized before averaging to ensure equal weight across different scales.Leakage-Free Scaling & Temporal Integration: Performed a Train-Test Split before applying StandardScaler to prevent data leakage. Unlike previous versions, the Year variable was included in the scaling process to correctly normalize temporal trends alongside other predictors.

Outcome of Work :
Optimized Predictive Power: Created highly correlated features, such as the education_income_interaction and disease_index, which provide a stronger mathematical foundation for high-accuracy model training than raw variables.Standardized & Integrity-Verified Data: Produced a clean environment where all numerical features (including Year) are standardized and data types are explicitly cast to float to eliminate system warnings and processing errors.Validated Master Datasets: Successfully exported two distinct, ready-to-use files—life_expectancy_train_master.csv and life_expectancy_test_master.csv—ensuring the evaluation phase is scientifically valid and free from leakage.

