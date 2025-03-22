# Titanic Survival Analysis

![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg)

## Overview

This repository contains a comprehensive machine learning project analyzing the Titanic disaster dataset to predict passenger survival. The analysis includes extensive exploratory data analysis (EDA), feature engineering, model building, and evaluation using multiple classification algorithms.

## Project Structure

- **Data Preprocessing**: Handles missing values, extracts features from text data, and encodes categorical variables
- **Feature Engineering**: Creates new features like family size, ticket information, and deck extraction
- **Exploratory Data Analysis**: Visualizes relationships between survival and various passenger attributes
- **Model Training**: Implements multiple models including Logistic Regression, Random Forest, Gradient Boosting, and SVM
- **Ensemble Method**: Combines individual models using a voting classifier for improved performance
- **Evaluation**: Assesses models using accuracy, precision, recall, F1 score, ROC curves, and precision-recall curves

## Key Features

- **Advanced Feature Engineering**
  - Title extraction from passenger names
  - Family size and type classification
  - Age bands and passenger class interactions
  - Deck extraction from cabin information
  - Ticket prefix analysis

- **Comprehensive Visualizations**
  - Survival rates by gender, class, age, and embarkation port
  - Interactive scatter plots showing relationships between multiple variables
  - Correlation heatmaps for feature importance analysis
  - Age-class survival rate matrix
  - Family size impact on survival

- **Model Performance Analysis**
  - Confusion matrices for all models
  - Feature importance visualization
  - ROC and precision-recall curves
  - Comparative performance metrics across models

## Requirements

```
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-analysis.git
cd titanic-survival-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python titanic_analysis.py
```

## Data

The analysis uses the famous Titanic dataset, which includes information about passengers such as:
- Passenger class (1st, 2nd, 3rd)
- Name, sex, and age
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Ticket number and fare
- Cabin number and embarkation port

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.82 | ~0.80 | ~0.70 | ~0.75 |
| Random Forest | ~0.84 | ~0.82 | ~0.75 | ~0.78 |
| Gradient Boosting | ~0.85 | ~0.83 | ~0.76 | ~0.79 |
| SVM | ~0.83 | ~0.81 | ~0.72 | ~0.76 |
| Ensemble | ~0.86 | ~0.84 | ~0.78 | ~0.81 |

*Note: Exact performance metrics may vary with different random seeds*

## Key Insights

- Female passengers had significantly higher survival rates than males
- Passengers in higher classes (1st, 2nd) were more likely to survive than those in 3rd class
- Children had higher survival rates, especially in 1st and 2nd class
- Passengers traveling alone had lower survival rates than those with family
- Certain passenger titles (like 'Mrs' and 'Miss') had higher survival rates
- Cabin deck information was a strong predictor of survival

## Visualization Examples

![Survival by Class and Gender](https://github.com/yourusername/titanic-survival-analysis/blob/main/images/class_gender_survival.png)

![Age Distribution](https://github.com/yourusername/titanic-survival-analysis/blob/main/images/age_distribution.png)

![Feature Importance](https://github.com/yourusername/titanic-survival-analysis/blob/main/images/feature_importance.png)

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Implementation of neural network models
- Additional feature engineering based on text mining of names and ticket information
- Handling of outliers and investigation of their impact on model performance
- Deployment as a web application for interactive predictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset is from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic)
- Inspired by various Kaggle kernels and tutorials on Titanic survival prediction
