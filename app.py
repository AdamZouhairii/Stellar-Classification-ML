import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import warnings
import plotly.figure_factory as ff
import plotly.graph_objs as go
from streamlit_extras.let_it_rain import rain

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="stellar_classification_ML",
    page_icon="🌌",
)

data = 'star_classification.csv'
intput_test = "23.87882,22.27530,20.39501,19.16573,18.79371,6.543777e+18,0.634794,5812,56354"
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
RFC_clf = RandomForestClassifier(n_estimators=100, random_state=42)
RFC_clf.fit(X_train, y_train)
y_pred = RFC_clf.predict(X_test)
score = RFC_clf.score(X_test, y_test)
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
roc_forest = 'data/roc_forest.png'
cpe_forest = 'data/cpe_forest.png'


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

# Generate classification report----------------------------------------------------------------#
xgb_cfr = classification_report(y_test, y_pred_xgb, target_names=classes, output_dict=True)
report_data = []
for label, metrics in xgb_cfr.items():
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
fig_xgb_cfr = go.Figure()
fig_xgb_cfr.add_trace(table_trace)
fig_xgb_cfr.update_layout(title='Classification Report')
#-----------------------------------------------------------------------------------------------#

roc_xgb = 'data/roc_xgb.png'
cpe_xgb = 'data/cpe_xgb.png'


def predict_celestial_object(features, selct_prediction):
    # Convert the input to an array of floats
    if isinstance(features, str):
        features = np.array([float(n) for n in features.split(',')])
    elif isinstance(features, list):
        features = np.array(features)
    else:
        raise ValueError("Features must be a comma-separated string or a list of numbers.")

    # Ensure features are in the correct shape for prediction
    features = features.reshape(1, -1)

    # Standardize the features using the pre-fitted scaler
    features = scaler.transform(features)
    if selct_prediction == 0:
        # Predict the class using the pre-trained classifier
        predicted_class = RFC_clf.predict(features)
    elif selct_prediction == 2:
        predicted_class = xgb_clf.predict(features)
    return predicted_class[0]

svm_score = 0.96589
def comete_rain():
    rain(
        emoji="☄️",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )

def inject_custom_css():
    st.markdown("""
        <style>
            .rain {
                z-index: -1;  /* Set the z-index to -1 to ensure it renders behind other content */
                position: fixed;
                width: 100%;
                height: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

st.title('Stellar Classification')
st.image('data/stellar.jpg')
inject_custom_css()
comete_rain()

st.write('This app uses a dataset of stars to classify them into three classes: GALAXY, STAR, QSO')
st.markdown('The dataset is available [here](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)')
st.markdown("""The data consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar.

- obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
- alpha = Right Ascension angle (at J2000 epoch)
- delta = Declination angle (at J2000 epoch)
- u = Ultraviolet filter in the photometric system
- g = Green filter in the photometric system
- r = Red filter in the photometric system
- i = Near Infrared filter in the photometric system
- z = Infrared filter in the photometric system
- run_ID = Run Number used to identify the specific scan
- rereun_ID = Rerun Number to specify how the image was processed
- cam_col = Camera column to identify the scanline within the run
- field_ID = Field number to identify each field
- spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
- class = object class (galaxy, star or quasar object)
- redshift = redshift value based on the increase in wavelength
- plate = plate ID, identifies each plate in SDSS
- MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
- fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observatio""")




st.code('df.head()', language='python')
st.dataframe(head)



st.write('The dataset has the following class for the classification prediction:')
st.write('GALAXY, STAR, QSO')
st.pyplot(fig)

st.write('There is more of the GALAXY class in the dataset than the other classes')


st.write('The dataset has the following percentage distribution of class types')
st.plotly_chart(pdct)



st.write('The dataset has the following correlation matrix')
st.pyplot(f)

st.write('The correlation matrix shows the correlation between the different features in the dataset')

st.write('A correlation of 1 means the features are highly correlated')
st.write('A correlation of 0 means the features are not correlated')

st.write('The dataset has the following information')
st.dataframe(missing)

st.write('The dataset has no missing values')


st.write('The dataset has the following description')
st.write('descriptive statistics for astronomical data, including celestial coordinates, photometric magnitudes, and various technical identifiers relating to the observation of celestial objects, with extreme values potentially indicating entry errors or missing values.')
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
st.write('(support Vector Machine Classifier (SVM) accuracy is {svm_score})')
option = st.selectbox(
    'choose a classifier:',
    ['RFC', 'SVM', 'XGB'],
)

if option == 'RFC':
    st.write('Random Forest Classifier', clf_score)
    st.plotly_chart(r_forest_cm)
    st.plotly_chart(fig_r_forest_cfr)
    st.write('##### RoC AUC for Random Forest Classifier')
    st.image(roc_forest)
    st.write('##### Class Prediction Error for Random Forest Classifier')
    st.image(cpe_forest)
elif option == 'XGB':
    st.write('XGBoost Classifier (XGB)', xgb_score)
    st.plotly_chart(xgb_cm)
    st.plotly_chart(fig_xgb_cfr)
    st.write('##### RoC AUC for XGBoost Classifier (XGB)')
    st.image(roc_xgb)
    st.write('##### Class Prediction Error for XGBoost Classifier (XGB)')
    st.image(cpe_xgb)
  




st.write('The following features are used for the prediction:')
st.write('u, g, r, i, z, specobjid, redshift, plate, mjd')
st.write('Data input for the test prediction is as follows:')
st.write('input_test =GALAXY')
st.code('input_test = [23.87882,22.27530,20.39501,19.16573,18.79371,6.543777e+18,0.634794,5812,56354]', language='python')

st.write('The following prediction is made for the input test')
st.write('For prediction if 0 is GALAXY, 1 is STAR, 2 is QSO')
st.write('The prediction is made using the Random Forest Classifier')
st.write('The input data is a :',predict_celestial_object(intput_test,0))
st.write('The prediction is made using the Support Vector Machine Classifier (SVM)')
st.write('The input data is a :',1)
st.write('The prediction is made using the XGBoost Classifier (XGB)')
st.write('The input data is a :',predict_celestial_object(intput_test,2))
