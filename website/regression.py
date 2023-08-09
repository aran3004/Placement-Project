import pandas as pd
import numpy as np
import shap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


def aggregate_shap_values(shap_values, one_hot_columns, original_features):
    # Create a dictionary to store aggregated SHAP values
    aggregated_shap = {
        original_feature: 0 for original_feature in original_features}

    print(aggregated_shap)
    # Iterate through one-hot encoded columns
    for column in one_hot_columns:
        # Extract the original feature name from the column name
        # Assuming the structure is "code_E09000002"
        original_feature = column.split('_')[0]
        print(original_feature)
        print(aggregated_shap[original_feature])

        # Accumulate SHAP values for the original feature
        aggregated_shap[original_feature] += shap_values[column].sum()

    return aggregated_shap


def hist_grad(df, target):
    # Removing all NaN values from target
    df = df.dropna(subset=target)
    # Calculate IQR for all numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Apply outlier removal using IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df[numerical_cols] < lower_bound) | (
        df[numerical_cols] > upper_bound)).any(axis=1)]

    numerical_cols = [
        col for col in df_cleaned._get_numeric_data().columns if col != target]

    categorical_cols = [
        col for col in df_cleaned.columns if col not in numerical_cols and col != target]

    # One-hot encode categorical features
    X = pd.get_dummies(df_cleaned.drop(
        [target] + numerical_cols, axis=1))
    X[numerical_cols] = df_cleaned[numerical_cols]  # Keep numerical columns
    y = df_cleaned[target]
    # 100 instances for use as the background distribution
    X100 = shap.utils.sample(X, 100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a regression model
    model = HistGradientBoostingRegressor()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return round(rmse, 2)


def hist_grad_with_shapley(df, target):
    # Removing all NaN values from target
    df = df.dropna(subset=target)
    # Calculate IQR for all numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Apply outlier removal using IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df[numerical_cols] < lower_bound) | (
        df[numerical_cols] > upper_bound)).any(axis=1)]

    numerical_cols = [
        col for col in df_cleaned._get_numeric_data().columns if col != target]

    categorical_cols = [
        col for col in df_cleaned.columns if col not in numerical_cols and col != target]

    # One-hot encode categorical features
    X = pd.get_dummies(df_cleaned.drop(
        [target] + numerical_cols, axis=1))
    X[numerical_cols] = df_cleaned[numerical_cols]  # Keep numerical columns
    y = df_cleaned[target]
    # 100 instances for use as the background distribution
    X100 = shap.utils.sample(X, 100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create a regression model
    model = HistGradientBoostingRegressor()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, X100)
    shap_values = explainer(X)

    # Aggregate SHAP values for original features
    original_features = [col for col in df.columns if col != target]
    # aggregated_shap = aggregate_shap_values(
    #     shap_values.values, X.columns, original_features)

    aggregated_shap = {
        original_feature: 0 for original_feature in original_features}

    shap_df = pd.DataFrame((zip(X.columns[np.argsort(np.abs(shap_values.values).mean(0))][::-1],
                                -np.sort(-np.abs(shap_values.values).mean(0)))),
                           columns=["feature", "importance"])
    print(shap_df)

    count = 0
    for i in range(len(shap_df)):
        for original_feature in original_features:
            if original_feature in shap_df.loc[i].feature:
                aggregated_shap[original_feature] += shap_df.loc[i].importance
                count += 1
    print(count)
    print(aggregated_shap)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return round(rmse, 2), aggregated_shap
