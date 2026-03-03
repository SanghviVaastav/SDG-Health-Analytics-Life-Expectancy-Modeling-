# SDG-Health-Analytics-Life-Expectancy-Modeling-
SDG-Health Analytics: Life Expectancy Modeling is a data science project that predicts life expectancy in developing countries using sanitation access and infant mortality data. The project applies data analysis and regression models to identify key health factors and support insights aligned with SDG 3 – Good Health and Well-Being.

This project predicts Life Expectancy in developing countries by analyzing the impact of sanitation access and infant mortality rates. Developed as part of the IIMSTC Internship, the study aligns with UN Sustainable Development Goal 3 (SDG 3) to promote Good Health and Well-Being.

The core objective is to move beyond simple correlation and build a predictive framework that quantifies how infrastructure (sanitation) and early-life health outcomes (mortality) dictate population longevity

Rafa Rahmath (1HK22CS114) - Multi Linear Regression Model

Task Explanation :
The primary objective of this project was to implement a predictive model using multi-linear regression to analyze a dataset and determine how various independent variables influence a target numerical output. The process involved extensive data preprocessing to handle potential statistical issues, such as multicollinearity, which was initially flagged by runtime warnings during the calculation of Variance Inflation Factors (VIF). To address these complexities and prevent overfitting, the script utilized a sophisticated machine learning pipeline that integrated data imputation and scaling with multiple regression techniques, specifically focusing on the performance of Ridge Regression.

Outcome :
The analysis successfully identified the Ridge Regression model as the most effective approach for this specific dataset, outperforming other tested methods. This model achieved a notable level of accuracy, yielding a Test R2 score of 0.8041, which suggests that approximately 80.4% of the variance in the target variable can be explained by the included features. To ensure the reliability of these results, the project concluded with a series of diagnostic visualizations—including "Actual vs. Predicted" and residual plots—which confirmed that the model maintained a strong fit and satisfied the underlying assumptions of linear regression.

Sruthi K S (4CB22CS136) - Data Collection and Preprocessing

Task Explanation
Collected the Life Expectancy dataset from Kaggle and performed complete data cleaning and preprocessing.
Handled missing values, removed duplicate records, filtered developing countries, and selected important features.
Applied outlier removal, log transformation, encoding, and normalization to prepare the dataset for model development.

Outcome of Work
Generated a fully cleaned and structured dataset ready for machine learning analysis.
Improved data quality by eliminating inconsistencies and scaling numerical features properly.
Saved the final processed dataset as life_expectancy_cleaned.csv for accurate model training and evaluation.

Saayanth M  (1ST22AI042) - End-to-End Data Preparation

Task Explanation:
Performed data preprocessing on the Life Expectancy dataset including handling missing values, encoding categorical variables, and feature scaling.
Conducted exploratory data analysis (EDA) to understand relationships between health, economic, and demographic factors affecting life expectancy.
Built and evaluated regression models to predict life expectancy based on selected features.

Task Outcomes:
Identified key factors such as income, immunization, and healthcare indicators significantly influencing life expectancy.
Successfully developed a predictive model with improved accuracy after proper preprocessing and feature engineering.
Gained hands-on experience in end-to-end data analytics workflow including data cleaning, visualization, modeling, and evaluation.

Vrushank Skanda B (1JT22AI059) - Feature Engineering 

Task Explanation :
Strategic Imputation: Identified null values with .isnull().sum() and implemented Median Imputation. This ensures the dataset remains robust against extreme GDP and mortality outliers that would otherwise skew a mean-based calculation.
Advanced Feature Engineering: Developed mathematically sound predictors including Healthcare Ratios and Composite Indices. Components were normalized via MinMaxScaler before averaging to prevent large-scale variables from dominating the index.
Leakage-Free Temporal Scaling: Executed a Train-Test Split prior to standardization to ensure model generalization. Unlike standard preprocessing, the Year variable was included in the StandardScaler pipeline to correctly normalize temporal trends.

Outcome of Work :
Maximized Feature Correlation: Engineered new variables, such as education_income_interaction, which exhibit significantly higher correlation with the target than raw data, providing a stronger foundation for high-accuracy regression.
Data Integrity & Type Safety: Produced a refined data environment where all features are standardized and explicitly cast to float, successfully eliminating pandas FutureWarning messages and dtype conflict errors.
Validated Master Datasets: Exported two distinct, ready-to-use master files—life_expectancy_train_master.csv and life_expectancy_test_master.csv—ensuring evaluation is scientifically valid and free from data leakage.

Sadiya Kulsum (1HK22CS124) - Linear Regression Model

Task Explanation:
The objective of this project was to build a predictive model using Linear Regression to analyze how multiple independent variables influence a continuous target variable. The workflow included data preprocessing steps such as handling missing values, detecting outliers, and checking multicollinearity using Variance Inflation Factor (VIF). A machine learning pipeline was implemented to integrate data imputation, feature scaling, and model training, ensuring a structured and reproducible process. The model was trained on a training dataset and evaluated on unseen test data to assess its generalization performance.

Outcome :
The Linear Regression model achieved a Test R² score of 0.85, explaining approximately 85% of the variance in the target variable. While minor multicollinearity effects were observed compared to regularized models, diagnostic plots—including Actual vs. Predicted and residual analysis—confirmed that key regression assumptions were reasonably satisfied. Overall, the model served as a strong baseline and provided clear interpretability of feature impacts on the target variable.

Ravi (1BO22EC048) - Model evaluation and its report

Task Explanation:
This project predicts life expectancy in developing nations using sanitation access and infant mortality.
Three models were implemented: Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression.
Among them, Simple Linear Regression achieved the best performance with the highest R² and lowest error values.

Outcome:
The study confirms that sanitation and infant mortality strongly influence life expectancy.
Simple Linear Regression provided the most accurate and reliable predictions.
The analysis supports data-driven public health decisions aligned with SDG 3 goals.
