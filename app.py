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
from sklearn.metrics import roc_curve,auc
import xgboost as xgb
import warnings
import plotly.figure_factory as ff
import plotly.graph_objs as go


warnings.filterwarnings("ignore", category=UserWarning)


data = r'C:\Users\Adame\Documents\GitHub\Stellar-Classification-ML\star_classification.csv'
df = pd.read_csv(data)
head = df.head()
info = df.info()
describe = df.describe()
missing = df.isnull().sum()
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

df = df.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID','fiber_ID'], axis = 1)
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

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Create annotated heatmap using Plotly
r_forest_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
r_forest_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))



# Generate classification report
r_forest_cfr = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

# Extract values from classification report
report_data = []
for label, metrics in r_forest_cfr.items():
    if label in classes:
        report_data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])

# Convert list to DataFrame
report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

# Create table trace with black background
table_trace = go.Table(
    header=dict(values=report_df.columns.tolist(), fill=dict(color='black'),  # Change header background color to blue
                font=dict(color='white'),  # Change header font color to white
                align='center'),
    cells=dict(values=[report_df['Class'], report_df['Precision'], report_df['Recall'], report_df['F1-Score'], report_df['Support']],
               fill=dict(color='black'),  # Change cell background color to black
               font=dict(color='white'),  # Change cell font color to white
               align='center'))

# Create figure
fig_r_forest_cfr = go.Figure()

# Add table trace to figure
fig_r_forest_cfr.add_trace(table_trace)

# Update layout
fig_r_forest_cfr.update_layout(title='Classification Report', title_x=0.5)




svm_clf = svm.SVC(kernel='rbf', C=1, random_state=0)
svm_clf.fit(X_train,y_train)
predicted = svm_clf.predict(X_test)
score = svm_clf.score(X_test, y_test)
svm_score_ = np.mean(score)


# Compute confusion matrix
cm = confusion_matrix(y_test, predicted)
# Create annotated heatmap using Plotly
svm_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
svm_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))



# Use XGBoost classifier with default parameters
xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train, eval_set=[(X_train, y_train)])
# Make predictions and evaluate the performance
y_pred_xgb = xgb_clf.predict(X_test)
xgb_score = xgb_clf.score(X_test, y_test)
xgb_score = np.mean(xgb_score)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
# Create annotated heatmap using Plotly
xgb_cm = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale='YlOrBr', showscale=True)
# Update layout
xgb_cm.update_layout(title='Confusion Matrix', xaxis=dict(title='Predicted label'), yaxis=dict(title='True label', autorange="reversed"))

st.set_page_config(
    page_title="stellar_classification_ML",
    page_icon="🌌",
)

st.title('Star Classification')
st.write('This app uses a dataset of stars to classify them into three classes: GALAXY, STAR, QSO')
st.write('The dataset contains 10000 rows and 18 columns')
st.code('df.head()', language='python')
st.dataframe(head)
st.write('The dataset has the following class for the classification prediction:')
st.write('GALAXY, STAR, QSO')
st.pyplot(fig)
st.write('The dataset has the following correlation matrix')
st.write('The correlation matrix shows the correlation between the different features in the dataset')
st.pyplot(f)
st.write('The dataset has the following information')
st.dataframe(info)
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
st.write('Support Vector Machine Classifier (SVM)', svm_score_)
st.plotly_chart(svm_cm)
st.write('XGBoost Classifier (XGB)', xgb_score)
st.plotly_chart(xgb_cm)
st.plotly_chart(fig_r_forest_cfr)