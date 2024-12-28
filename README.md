## End to End ML project (Income Prediction)

### Step 1: Setting up github repository

git init

git add <file_name>

 git commit -m "<commit_message>"

git branch -M main

git remote add origin "<github_repository_link>"

git push -u origin main


### If doing for the first time. Ensure config:

git config --global user.name "John Doe"

git config --global user.email johndoe@example.com

### To check config:

git config --global user.name

git config --global user.email

### If files exists in github repsoitory: 

git pull origin main (to pull code to your VS code)

### For regular push of code:

git add .

git commit -m "<commit_message>"

git push origin main

## Life Cycle of Machine Learning Project:

### 1. Understanding the Problem Statement

### 2. Data Collection

### 3. Data Checks to Perform

### 4. Exploratory Data Analysis (EDA)

### 5. Conclusions from EDA

### 6. Data Pre-Processing II

### 7. Model Training

### 8. Choose Best Model


### Project Goal:

The goal is to predict whether a person has an income of more than 50K a year or not. This is a binary classification problem where a person is classified into the >50K group or the <=50K group.

### Dataset Source:

[Adult Census Dataset](https://www.kaggle.com/datasets/overload10/adult-census-dataset)

### Dataset Features:

age: Represents the age of the person

workclass: Represents the sector in which a person is working

education: Represents the level of education acquired by a person

marital status: Represents a person's marital status

occupation: Represents the occupation a person is working as

relationship: Represents their family relationship

race: Represents the race of the person

sex: Represents the gender of the person

capital gain: Represents the capital gains a person is making

capital loss: Represents the capital losses a person is incurring

hours per week: Represents the number of hours a person is working

country: Represents the country in which a person is staying

salary: Represents whether the income of the person is above or below 50K

### Data Checks to Perform

Check Missing Values

Check Duplicates

Check Data Type

Check the Number of Unique Values of Each Column

Check Statistics of the Dataset

Check Various Categories Present in Different Categorical Columns

Feature Engineering

### Exploratory data analysis

Created an ipynb file for EDA and handling data cleaning.

### Data Preprocessing II

Create another ipynb file called as Model_Training for doing data preprocessing part II and model training and finding best model.

### Model training and choosing best model:

Training multiple classification models and finding best accuracies. Choosing the best classification model based on best accuracy in training set and testing set. 

Obtained the best model as Gradient boosting classifier with training accuracy as 86.76 percent and testing accuracy as 86.86 percent. Performed hyperparameter tuning as well to enhance the performance of gradient boosting classifier. 

### Saving_models folder

Create pickle folder under saving_models folder containing train.py, pred.py and pickle files like encoder.pkl(for encoding categorical features), scaler.pkl(standard scaling the numerical features), model.pkl(for predicting output using model)

### Created deployment Folder

Added app.py for deployment

Added template Folder with home.html for the front-end interface

### Result

Created a machine learning model of Gradient Boosting Classifier with training and testing accuracies of 86.76 and 87.33 percent and deployed it in form of web application using Flask web framework.

