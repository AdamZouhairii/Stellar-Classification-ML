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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import warnings
import plotly.figure_factory as ff
import plotly.graph_objs as go


warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="stellar_classification_ML",
    page_icon="🌌",
)

data = r'C:\Users\Adame\Documents\GitHub\Stellar-Classification-ML\star_classification.csv'
df = pd.read_csv(data)
head = df.head()
describe = df.describe()
missing = df.isnull().sum()
class_counts = df["class"].value_counts()


# Plotting the value counts of classes --------------------------------#
fig, ax = plt.subplots(facecolor='black')                              #
ax.bar(class_counts.index, class_counts.values, color='red')           #
ax.grid(False)                                                         #
ax.set_xlabel('Class', color='white')                                  #
ax.set_ylabel('Count', color='white')                                  #
ax.set_title('Value Counts of Classes', color='white')                 #
ax.set_facecolor('black')                                              #
ax.tick_params(axis='x', colors='white', rotation=45)                  #
ax.tick_params(axis='y', colors='white')                               #
ax.set_xticks(range(len(class_counts.index)))                          #
ax.set_xticklabels(class_counts.index, rotation=45, ha='right')        #
# ---------------------------------------------------------------------#

# Percentage distribution of class types ------------------------------------------------------#
colors = ['#48678b', '#748db4', '#b3c1d2']
pull = [0.015, 0.025, 0.025] 
pdct = go.Figure(data=[go.Pie(labels=class_counts.index,
                             values=class_counts.values,
                             pull=pull,
                             marker_colors=colors,
                             textinfo='percent+label',
                             textfont=dict(family='Times New Roman', size=12),
                             )])
pdct.update_layout(title_text='Percentage Distribution of Class Types',
                  title_font=dict(family='Times New Roman', size=16),
                  )
#-----------------------------------------------------------------------------------------------#

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

# Correlation Matrix --------------------------------------------------------------------------#
df = df.drop(['rerun_ID'], axis=1)
corr = df.corr()
f, ax = plt.subplots(figsize=(13, 9))
ax.set_facecolor('black')
sns_heatmap = sns.heatmap(corr, cmap="YlOrBr", annot=True, linewidths=0.5, fmt='.2f', ax=ax)
f.set_facecolor('black')
ax.set_xlabel(ax.get_xlabel(), color='white')
ax.set_ylabel(ax.get_ylabel(), color='white')
ax.set_xticklabels(ax.get_xticklabels(), color='white')
ax.set_yticklabels(ax.get_yticklabels(), color='white')
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
ax.set_title('Correlation Matrix', color='white')       
#-----------------------------------------------------------------------------------------------#


df = df.drop(['obj_ID','alpha','delta','run_ID','cam_col','field_ID','fiber_ID'], axis = 1)
x = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

# Using Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
clf_score = np.mean(score)

classes = ["GALAXY", "STAR", "QSO"]

# Compute confusion matrix --------------------------------------------------------------------#
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Create annotated heatmap using Plotly
r_forest_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
r_forest_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))
#-----------------------------------------------------------------------------------------------#

# Generate classification report----------------------------------------------------------------#
r_forest_cfr = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
report_data = []
for label, metrics in r_forest_cfr.items():
    if label in classes:
        report_data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
table_trace = go.Table(
    header=dict(values=report_df.columns.tolist(), fill=dict(color='black'),  # Change header background color to blue
                font=dict(color='white'),  # Change header font color to white
                align='center'),
    cells=dict(values=[report_df['Class'], report_df['Precision'], report_df['Recall'], report_df['F1-Score'], report_df['Support']],
               fill=dict(color='black'),  # Change cell background color to black
               font=dict(color='white'),  # Change cell font color to white
               align='center'))
fig_r_forest_cfr = go.Figure()
fig_r_forest_cfr.add_trace(table_trace)
fig_r_forest_cfr.update_layout(title='Classification Report')
#-----------------------------------------------------------------------------------------------#
roc_forest = 'roc_forest.png'
cpe_forest = 'cpe_forest.png'


svm_clf = svm.SVC(kernel='rbf', C=1, random_state=0)
svm_clf.fit(X_train,y_train)
predicted = svm_clf.predict(X_test)
score = svm_clf.score(X_test, y_test)
svm_score_ = np.mean(score)


# Compute confusion matrix---------------------------------------------------------------------#
cm = confusion_matrix(y_test, predicted)
# Create annotated heatmap using Plotly
svm_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
svm_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))
#-----------------------------------------------------------------------------------------------#

roc_svm = 'roc_svm.png'
cpe_svm = 'cpe_svm.png'


# Use XGBoost classifier with default parameters
xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train, eval_set=[(X_train, y_train)])
# Make predictions and evaluate the performance
y_pred_xgb = xgb_clf.predict(X_test)
xgb_score = xgb_clf.score(X_test, y_test)
xgb_score = np.mean(xgb_score)

# Compute confusion matrix---------------------------------------------------------------------#
cm = confusion_matrix(y_test, y_pred_xgb)
# Create annotated heatmap using Plotly
xgb_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
xgb_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))
#-----------------------------------------------------------------------------------------------#

roc_xgb = 'roc_xgb.png'
cpe_xgb = 'cpe_xgb.png'

"""
   Stellar Classification ML App

"""

st.title('Star Classification')
st.write('This app uses a dataset of stars to classify them into three classes: GALAXY, STAR, QSO')
st.write('The dataset contains 10000 rows and 18 columns')
st.code('df.head()', language='python')
st.dataframe(head)
st.write('The dataset has the following class for the classification prediction:')
st.write('GALAXY, STAR, QSO')
st.pyplot(fig)
st.write('The dataset has the following percentage distribution of class types')
st.plotly_chart(pdct)
st.write('The dataset has the following correlation matrix')
st.write('The correlation matrix shows the correlation between the different features in the dataset')
st.pyplot(f)
st.write('The dataset has the following information')
st.write('The dataset has no missing values')
st.dataframe(missing)
st.write('The dataset has the following description')
st.dataframe(describe)
st.write('The dataset has been cleaned and outliers have been removed')
st.write('15256 outliers have been removed')
st.write('The dataset has been split into training and testing datasets')
st.write('0.67 of the dataset has been used for training and 0.33 for testing')
st.write('The following classifiers have been used:')
st.write('Random Forest Classifier','Support Vector Machine Classifier (SVM)','XGBoost Classifier (XGB)' )
st.write('The following metrics have been used to evaluate the classifiers:')
st.write('Confusion Matrix', 'Classification Report', 'ROC AUC', 'Class Prediction Error')
st.write('The accuracy of the classifiers is as follows:')
st.write('Random Forest Classifier', clf_score)
st.plotly_chart(r_forest_cm)
st.plotly_chart(fig_r_forest_cfr)
st.write('##### RoC AUC for Random Forest Classifier')
st.image(roc_forest)
st.write('##### Class Prediction Error for Random Forest Classifier')
st.image(cpe_forest)
st.write('Support Vector Machine Classifier (SVM)', svm_score_)
st.plotly_chart(svm_cm)
st.write('##### RoC AUC for Support Vector Machine Classifier (SVM)')
st.image(roc_svm)
st.write('##### Class Prediction Error for Support Vector Machine Classifier (SVM)')
st.image(cpe_svm)
st.write('XGBoost Classifier (XGB)', xgb_score)
st.plotly_chart(xgb_cm)
st.write('##### RoC AUC for XGBoost Classifier (XGB)')
st.image(roc_xgb)
st.write('##### Class Prediction Error for XGBoost Classifier (XGB)')
st.image(cpe_xgb)