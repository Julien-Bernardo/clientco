# clientco

# EDA
We explored the data and performed preliminary preprocessing and visualizations to gain insights. We found that we could classify clients in three categories (high-value, average, and low-value) using an unsupervised clustering algorithm K-means. Additionally, we also found that products could also be split in three categories with K-means (very profitable with low delivery time, high delivery time, less profitable).
```
notebooks/EDA_JFB.ipynb
notebooks/EDA_HG.ipynb
```

# Preprocessing
We apply preprocessing steps to the raw data with a script with modular functions. We also defined churned clients as those that haven't ordered in mean + 1*standard deviation days. The processed data allows us to predict churn with a ML classifier.
```
src/preprocessing/preprocess.py
```

# Churn Prediction
We used Pycaret that automatically finds the best algorithm for churn classification. We also fine tune it with incredible results (AUC=0.9995). This could be due to the fact that only 11% of the clients churn leaving an imbalance in the target classes. 
```
notebooks/churn_prediction.ipynb
```

# Next setps:
- Fix the imbalance in target classes (churn vs no churn)
- Develop a web app for interaction with the data and insightful visualizations