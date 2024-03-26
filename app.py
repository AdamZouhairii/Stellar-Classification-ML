import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
import xgboost as xgb

data = 'star_classification.csv'
df = pd.read_csv(data)
head = df.head()
class_counts = df["class"].value_counts()

# Creating a new figure and axis
fig, ax = plt.subplots()

# Plotting a bar chart
ax.bar(class_counts.index, class_counts.values, color='red')

# Setting labels and title
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Value Counts of Classes')

df["class"] = [0 if i == 'GALAXY' else 1 if i == 'STAR' else 2 for i in df["class"]]

lof = LocalOutlierFactor()
y_pred = lof.fit_predict(df)
x_score = lof.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

# threshold
threshold = -1.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()
df.drop(outlier_index, inplace = True)


corr = df.corr()
f, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(corr, cmap="YlOrBr" , annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
ax.set_title('Correlation Matrix')

st.title('Star Classification')
st.write('This app uses a dataset of stars to classify them into three classes: GALAXY, STAR, QSO')
st.write('The dataset contains 10000 rows and 18 columns')
st.code('df.head()', language='python')
st.dataframe(head)
st.write('The dataset has been cleaned and outliers have been removed')
st.write('15256 outliers have been removed')
st.write('The dataset has been split into training and testing datasets')
st.write('0.67 of the dataset has been used for training and 0.33 for testing')
st.write('The following classifiers have been used:')
st.write('Random Forest Classifier')
st.write('Support Vector Machine Classifier (SVM)')
st.write('XGBoost Classifier (XGB)')
st.write('The following metrics have been used to evaluate the classifiers:')
st.write('Confusion Matrix')
st.write('Classification Report')
st.write('ROC AUC')
st.write('Class Prediction Error')
st.write('The accuracy of the classifiers is as follows:')
st.pyplot(fig)
st.pyplot(f)