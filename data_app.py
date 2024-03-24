import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC
from yellowbrick.style import set_palette
import xgboost as xgb

data = r'C:\Users\Adame\Documents\GitHub\Stellar-Classification-ML\star_classification.csv'
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