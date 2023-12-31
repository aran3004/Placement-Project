from flask import Blueprint, session, redirect, url_for, render_template, flash
from . import db
from flask_login import current_user
from .models import Datasets, Features, User, Log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import pandas as pd
import numpy as np
from .regression import *

search_algo = Blueprint('search_algo', __name__)

# Load a pre-trained word embedding model
nlp = spacy.load("en_core_web_md")

# finds others features' that could match the current user


@search_algo.route('/search')
def search():
    # Find all tasks current user has uploaded
    datasets = Datasets.query.all()
    dataset_descriptions = {}
    for dataset in datasets:
        if dataset.user_id == current_user.id:
            dataset_descriptions[dataset.id] = dataset.task

    # Find all features that are uploaded by everyone except the current user
    features = Features.query.all()
    feature_descriptions = {}
    for feature in features:
        if feature.user_id != current_user.id:
            feature_descriptions[feature.id] = feature.info

    # For cases of no uploaded features or ML tasks
    if len(feature_descriptions) == 0:
        return 'There are currently no features available'
    elif len(dataset_descriptions) == 0:
        return 'Please upload a dataset and task to run the search algorithm'
    else:
        # Preprocessing the descriptions
        dataset_descriptions = preprocess(dataset_descriptions)
        feature_descriptions = preprocess(feature_descriptions)

        # Compute word embeddings for the descriptions
        dataset_embeddings = {
            id: nlp(description).vector for id, description in dataset_descriptions.items()}
        feature_embeddings = {
            id: nlp(description).vector for id, description in feature_descriptions.items()}

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(
            list(dataset_embeddings.values()), list(feature_embeddings.values()))

        # Match datasets with features and their cosine similarity values
        results = {}
        for i, (dataset_id, dataset_description) in enumerate(dataset_descriptions.items()):
            similarities = similarity_matrix[i]
            # Sort indices in descending order
            ranked_indices = similarities.argsort()[::-1]
            ranked_feature_ids = [list(feature_embeddings.keys())[idx]
                                  for idx in ranked_indices]
            ranked_similarities = [similarities[idx] for idx in ranked_indices]

            print(
                f"Dataset ID: {dataset_id}, Description: {dataset_description}")
            print("Ranked features:")
            results[f"Dataset ID: {dataset_id}"] = {}

            for feature_id, similarity in zip(ranked_feature_ids, ranked_similarities):
                if similarity > 0.6:

                    dataset_file_path, model_type = get_dataset_by_id(
                        dataset_id)
                    feature_file_path = get_feature_by_id(feature_id)

                    dataset_df = pd.read_csv(dataset_file_path)
                    feature_df = pd.read_csv(feature_file_path)
                    dataset_df = preprocess_df_strings(dataset_df)
                    feature_df = preprocess_df_strings(feature_df)

                    matching_columns = get_matching_columns(
                        dataset_df, feature_df)

                    for a, b in enumerate(matching_columns):
                        print(b[0], b[1], matching_columns[b])
                    # print(a,b for a,b in enumerate(matching_columns))
                    # columns, matching_score = matching_columns.items()
                    # column1, column2 = columns

                    feature_description = feature_descriptions[feature_id]
                    results[f"Dataset ID: {dataset_id}"][f'Feature_ID: {feature_id}'] = {
                        'Description': feature_description, 'Similarity': str(similarity), 'Matching Column 1': b[0], 'Matching Column 2': b[1], 'Matching Score': matching_columns[b]}
                    print(
                        f"- Feature ID: {feature_id}, Description: {feature_description}, Similarity: {similarity}, Matching Column 1: {b[0]}, Matching Column 2: {b[1]}, Matching Score: {matching_columns[b]}")

            if len(results[f"Dataset ID: {dataset_id}"]) == 0:
                print("No available features for this dataset. Try adding more tags or adjusting the description. Visit your profile to do so.")

            print("-----")

        session['results'] = results

        return redirect(url_for('email.feature_match'))


@search_algo.route('/match')
def match():
    task_feature_groups = datasets_and_feature_groups()

    # Need to remove features that are not suitbale based on descritpion and info
    # Store these scores in an array and remove features from array if they are too low in similarity

    print(task_feature_groups)

    # Dictionary to store the matched pairs
    matched_pairs = {}

    # Compute word embeddings for the descriptions
    task_embeddings = {}  # Store task embeddings outside the loop
    feature_embeddings = {}  # Store feature embeddings outside the loop

    # Compute word embeddings for the descriptions
    for task in task_feature_groups:
        task_embeddings[task] = nlp(task.task).vector
        for feature in task_feature_groups[task]:
            feature_embeddings[feature.feature_name] = nlp(feature.info).vector

    # Loop through each task
    for task, task_embedding in task_embeddings.items():
        # Initialize a list to store the similarity scores for this task
        similarity_scores = []

        # Loop through each feature and its embedding
        for feature, feature_embedding in feature_embeddings.items():
            # Calculate cosine similarity between the task and feature embeddings
            similarity_score = cosine_similarity(
                [task_embedding], [feature_embedding])[0][0]

            # Add the similarity score to the list
            similarity_scores.append((feature, similarity_score))

        # Sort the similarity scores in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Set a similarity threshold (adjust this value based on your needs)
        similarity_threshold = 0.4

        # Find features with similarity scores above the threshold
        matched_features = []
        for feature, score in similarity_scores:
            if score >= similarity_threshold:
                for feature_obj in task_feature_groups[task]:
                    if feature_obj.feature_name == feature:
                        matched_features.append(feature_obj)

        # Store the matched features in the dictionary
        matched_pairs[task] = matched_features

    users_and_features = []

    # Print the matched pairs
    for task, matched_features in matched_pairs.items():
        print(f"Task: {task}")
        task_df = pd.read_csv(task.file_path)
        merged_df = pd.read_csv(task.file_path)
        print(task_df.columns)
        features_for_analysis = []
        for feature_obj in matched_features:
            print(
                f"Matched Feature: {feature_obj}, Feature Name: {feature_obj.feature_name}, User ID: {feature_obj.user_id}")
            feature_df = pd.read_csv(feature_obj.file_path)
            matches = get_matching_columns(task_df, feature_df)
            if len(matches) == 0:
                print('This features is not a match for this task')
            elif list(matches.values())[0] < 60:
                print(
                    "This feature is a match, but doesn't cover enough of the original dataset")
            else:
                features_for_analysis.append(feature_obj)
                column1, column2 = list(matches.keys())[
                    0][0], list(matches.keys())[0][1]
                print(
                    f'Matching Column 1 is: {column1}, Matching Column 2 is: {column2} ')
                features_list = feature_df.columns.values.tolist()
                print(
                    f'Columns in Feature: {features_list}')
                for col in features_list:
                    if col != column2:
                        print(f'Added Feature Column Name:{col}')
                        users_and_features.append((feature_obj.user_id, col))

                print(f'Matching Rate: {list(matches.values())[0]}')
                merged_df = pd.merge(merged_df, feature_df,
                                     left_on=column1, right_on=column2)
                if column1 != column2:
                    merged_df.drop(column2, inplace=True, axis=1)
            print('----------------------')

        print(users_and_features)
        if task.model_type != 'regression':
            print("Models other than regression are still yet to be coded")

        if len(features_for_analysis) > 0:
            print(merged_df.head())
            print(merged_df.columns)
            if task.model_type == 'regression':
                original_dataset_result = hist_grad(task_df, task.target)
                merged_dataset_result, aggregated_shap = hist_grad_with_shapley(
                    merged_df, task.target)

            retained_dataset = round(
                len(merged_df.index)/len(task_df.index)*100, 2)
            print(
                f'Result from original task alone: {original_dataset_result}')
            print(
                f'Dataset Retained: {retained_dataset}%')
            print(f'Result from added features: {merged_dataset_result}')
            if merged_dataset_result < original_dataset_result:
                percentage_improvement = round(
                    ((original_dataset_result - merged_dataset_result)/original_dataset_result)*100, 2)
                print(
                    f'Percentage Improvement to Model: {percentage_improvement}%')
                # lets assume for now that the public bid is for 5% improvement in model
                print(task.public_bid)
                to_pay = task.public_bid*percentage_improvement/5
                print(f'Credit to be distributed: {to_pay}')

            added_feature_importance = {}
            for feature in users_and_features:
                user = feature[0]
                column_name = feature[1]
                if column_name in aggregated_shap:
                    if added_feature_importance.get(user) is not None:
                        added_feature_importance[user] = added_feature_importance[user] + \
                            aggregated_shap[column_name]
                    else:
                        added_feature_importance[user] = aggregated_shap[column_name]
            print(added_feature_importance)
            sum_of_feature_importance = sum(added_feature_importance.values())
            payment_distibution = {}
            for user in added_feature_importance:
                if added_feature_importance[user] != 0:
                    payment_distibution[user] = round((
                        added_feature_importance[user]/sum_of_feature_importance) * to_pay)
            print(payment_distibution)
            for payee in payment_distibution:
                payee_object = User.query.get(payee)
                payee_object.credit = payee_object.credit + \
                    payment_distibution[payee]

                log_description = f'Received <strong> {payment_distibution[payee]} credit </strong> for contributing to Task: <strong> {task.dataset_name} </strong>'
                new_log = Log(description=log_description, user_id=payee)
                db.session.add(new_log)

                db.session.commit()
                flash('Credit Paid', category='success')

            # Take payment from user with task
            paying_user = task.user_id
            paying_user = User.query.get(paying_user)
            paying_user.credit = paying_user.credit - round(to_pay)
            log_description = f'Paid <strong> {to_pay} credit </strong> for adding {len(payment_distibution)} features to dataset: <strong>{task.dataset_name}</strong>. <br> Model improved from {original_dataset_result} to {merged_dataset_result}, which is equal to <strong> {percentage_improvement}% improvement</strong>. <br>Percentage of original dataset retained:<strong> {retained_dataset}%. </strong>'
            new_log = Log(description=log_description, user_id=paying_user.id)
            db.session.add(new_log)
            db.session.commit()
            flash('Credit Withdrawn', category='success')
        else:
            print('There are no suitable features for this task at this time')

        print('------------------------------------------------------------------------------------------')

    # Need to get the merge ratings and put in an array
    # At some point i will make a table on the profile page that allows the user to test features for their task and compare rating
    return render_template('home.html', user=current_user)

# Preprocess descriptions (cleaning and tokenization)


def preprocess_description(description):
    # Remove punctuation
    description = description.translate(
        str.maketrans('', '', string.punctuation))
    # Lowercasing
    description = description.lower()
    # Tokenization and stopwords removal
    tokens = nlp(description)
    processed_tokens = [
        token.text for token in tokens if token.text not in STOP_WORDS]
    # Join the processed tokens back into a string
    processed_description = ' '.join(processed_tokens)
    return processed_description


def preprocess(descriptions):
    processed_descriptions = {}
    for id, description in descriptions.items():
        # Remove punctuation
        description = description.translate(
            str.maketrans('', '', string.punctuation))
        # Lowercasing
        description = description.lower()
        # Tokenization and stopwords removal
        tokens = nlp(description)
        processed_tokens = [
            token.text for token in tokens if token.text not in STOP_WORDS and token.text not in ['data', 'dataset']]
        # Join the processed tokens back into a string
        processed_description = ' '.join(processed_tokens)
        processed_descriptions[id] = processed_description
    return processed_descriptions


def preprocess_df_strings(df):
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].apply(lambda x: x.str.lower())
    return df

# Find columns that have matching items, even if the feature label is different


def get_matching_columns(df1, df2):
    df1 = preprocess_df_strings(df1)
    df2 = preprocess_df_strings(df2)
    cols1 = df1.columns
    cols2 = df2.columns

    matching_columns = {}

    for col1 in cols1:
        for col2 in cols2:
            unique_features_df1 = set(df1[col1].unique())
            unique_features_df2 = set(df2[col2].unique())

            matching_features = unique_features_df1.intersection(
                unique_features_df2)
            matching_features_df = df1[df1[col1].isin(matching_features)]
            # How much of original dataset can be matched up to
            matching_rate = (len(matching_features_df) / len(df1)) * 100

            # Matching rate above 10%
            if matching_rate > 10:
                matching_columns[(col1, col2)] = matching_rate

    return matching_columns


def get_dataset_by_id(dataset_id):
    dataset = Datasets.query.get(dataset_id)

    if dataset is None:
        return 'error: Dataset not found', 404

    file_path = dataset.file_path
    model_type = dataset.model_type

    return file_path, model_type


def get_feature_by_id(feature_id):
    feature = Features.query.get(feature_id)

    if feature is None:
        return 'error: Feature not found', 404

    file_path = feature.file_path

    return file_path


def datasets_and_feature_groups():
    # Find all tasks uploaded
    datasets = Datasets.query.all()

    # Find all features that are uploaded
    features = Features.query.all()

    # Create a dictionary to store the matched datasets and features
    task_feature_groups = {}

    # Loop through each dataset and find the matching features based on user_id
    for dataset in datasets:
        dataset.task = preprocess_description(dataset.task)
        # Filter features that do not have the same user_id as the dataset
        matching_features = []
        for feature in features:
            if feature.user_id != dataset.user_id:
                feature.info = preprocess_description(feature.info)
                matching_features.append(feature)

        # Add the matched features to the dataset in the dictionary
        task_feature_groups[dataset] = matching_features

    return task_feature_groups
