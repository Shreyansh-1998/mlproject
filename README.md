# End to End Machine Learning Project- Performance Indicator

## Introduction About the data:

*The goal is to predict the performance of student's score using(Regression Analysis)*

Target variable:
* `math score`: Marks obtained in Mathematics

Datasource link :[https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?select=StudentsPerformance.csv]

# Approach for the project 

1. EDA:
    * Created a jupyter notebook to dive into the project deeply and conduct an Exploratory Data Analysis


2. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

3. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

4. Model Training : 
    * In this phase base model is tested . The best model found was Linear Regression
    * After this hyperparameter tuning is performed on every model.
    * This model is saved as pickle file.



# Guide to run the project:

### 1.Clone the project using
    ```
    git clone https://github.com/Shreyansh-1998/mlproject.git
    ```
### 2. Install all the necessary libraries with 
  ```
  pip install -r requirements.txt
  ```
### 3. Run the application 
``` 
python3 app.py
```
### Future Plan

1. Build a prediction pipeline which converts given data into dataframe and has various functions to load pickle files and predicts the final results.

2. Additionaly, icing on the cake would be to build a flask application and host it on AWS.

###Credits to KRISH NAIK:[https://github.com/krishnaik06]

