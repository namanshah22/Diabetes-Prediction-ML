# Diabetes Prediction

This project aims to predict whether a person is diabetic or not based on several health-related features using machine learning techniques.

## Dataset

The dataset used for this project is stored in the file `diabetes.csv`. It contains information about various health measurements for a group of individuals, along with an outcome label indicating whether they are diabetic or not.

## Getting Started

To run this project, follow the steps below:

1. Clone the repository: git clone https://github.com/namanshah22/Diabetes-Prediction-ML 
2. Install the required dependencies: pip install pandas numpy sklearn
3. Run the project


## Preprocessing

The dataset is preprocessed before training the machine learning model. The following steps are performed:

- Load the dataset using `pandas`.
- Split the dataset into features (`x`) and the target variable (`y`).
- Standardize the features using `StandardScaler` from `sklearn.preprocessing`.

## Model Training

A support vector machine (SVM) model with a linear kernel is used for training the classifier. The following steps are performed:

- Split the preprocessed data into training and testing sets using `train_test_split` from `sklearn.model_selection`.
- Initialize the SVM classifier with a linear kernel.
- Fit the classifier to the training data using the `fit` method.
- Predict the labels for the training data and calculate the accuracy score using `accuracy_score` from `sklearn.metrics`.

## Model Evaluation

The trained SVM model is evaluated on the testing data to assess its performance. The following steps are performed:

- Predict the labels for the testing data using the trained classifier.
- Calculate the accuracy score of the model on the testing data using `accuracy_score`.

## Prediction

You can make predictions on new data by providing the input sample. The following steps are performed:

- Prepare an input sample as a Python list or numpy array.
- Reshape the input sample to match the shape expected by the scaler and classifier.
- Transform the input sample using the pre-fitted scaler.
- Predict the label for the transformed sample using the trained classifier.
- Print the prediction result based on the predicted label.

## Conclusion

This project demonstrates the use of a support vector machine model for diabetes prediction based on health-related features. The trained model achieves a certain accuracy on the testing data. Further improvements and optimizations can be explored to enhance the accuracy and generalizability of the model.
