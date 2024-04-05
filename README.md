# Stellar Data Classification Project

## Project Overview

This project aims to classify stellar objects into three categories: galaxies, stars, and quasars (QSO). To achieve this, we apply three different machine learning models: Random Forest, Support Vector Machine (SVM), and XGBoost. These models were chosen for their effectiveness in handling complex datasets with high accuracy.

## Data

The [dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) used in this project contains various features of stellar objects that are crucial for their classification. The features include but are not limited to, spectral data, magnitudes, and distances.

## Models

- **Random Forest Classifier**: An ensemble learning method for classification that operates by constructing a multitude of decision trees at training time. It outputs the class that is the mode of the classes of the individual trees.
  
- **Support Vector Machine (SVM)**: A powerful and versatile machine learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection. It is particularly well-suited for classification of complex but small- or medium-sized datasets.
  
- **XGBoost**: An implementation of gradient boosted decision trees designed for speed and performance. It is a highly efficient and scalable model that has proven to achieve remarkable accuracy in various machine learning competitions.

## Requirements

Please ensure you have the following packages installed to run the project:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- yellowbrick
- xgboost
- plotly
- streamlit

You can install all required packages using the provided `requirements.txt` file.

## Installation

To set up your environment to run this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

To run the project, simply launch the `app.py` script. This script initiates the classification process using the specified machine learning models and show graph of the stellar data.

### Running the Application

Execute the following command in your terminal:

```bash
python app.py
```
Ensure you are in the project's root directory before executing this command. The app.py script encompasses the workflow for data preprocessing, model training, and evaluation, outputting the classification results for the stellar data.
## Results

The performance of each model is evaluated based on accuracy, precision, recall, and F1 score. Additionally, we utilize confusion matrices, classification reports, and ROC curves for a comprehensive analysis of the models' effectiveness in classifying stellar data.

Visualizations provided by Plotly, Yellowbrick and matplotlib help in understanding the model performance better through graphical representations.

## Analysis

Detailed analysis of each model's performance should be included here, comparing the effectiveness of Random Forest, SVM, and XGBoost in classifying the stellar data. Insights on feature importance, model overfitting or underfitting, and strategies for model improvement can also be discussed.

## Conclusion

This project demonstrates the application of machine learning models to classify stellar objects into galaxies, stars, and QSO. The comparative analysis provides insights into the strengths and weaknesses of each model in handling astronomical data. Future directions could include exploring more complex models or deep learning approaches to improve classification accuracy.

## How to Contribute

Contributions to the project are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a detailed description of your changes.

Please ensure your code adheres to the project's coding standards and include tests where applicable.

## License

This project is open source and available under the [MIT License](LICENSE).
