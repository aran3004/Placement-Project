from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def logistic_regression_model(dataframe, target):
    # Split the dataframe into features (X) and target variable (y)
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Perform one-hot encoding for categorical columns
    ct = ColumnTransformer(
        [('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
    X = ct.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    accuracy = model.score(X_test, y_test)
    accuracy = round(accuracy, 4)

    return model, accuracy
