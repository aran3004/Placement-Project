from flask import Blueprint, session, redirect, url_for, render_template
from . import db
from flask_login import current_user
from .models import Datasets, Features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import pandas as pd
import numpy as np

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
                    dataset_df = preprocess_df(dataset_df)
                    feature_df = preprocess_df(feature_df)

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

    # Seeing datasets and the other features
    print(task_feature_groups)
    for task in task_feature_groups:
        print("--------")
        print(task)
        print(task.dataset_name)
        print(task.task)
        for feature in task_feature_groups[task]:
            print(
                f"User ID:{feature.user_id} || {feature.feature_name} || {feature.id} || {feature.file_path} || {feature.info}")
        print("--------")

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

    # Print the matched pairs
    for task, matched_features in matched_pairs.items():
        print(f"Task: {task}")
        task_df = pd.read_csv(task.file_path)
        for feature_obj in matched_features:
            print(
                f"Matched Feature: {feature_obj}, Feature Name: {feature_obj.feature_name}, User ID: {feature_obj.user_id}")
            feature_df = pd.read_csv(feature_obj.file_path)
            print(task_df.head())
            print(feature_df.head())
            print(get_matching_columns(task_df, feature_df))

    # Need to get the merge ratings and put in an array
    # At some point i will make a table on the profile page that allows the user to test features for their task and compare rating
    return render_template('profile.html', user=current_user)

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


def preprocess_df(df):
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].apply(lambda x: x.str.lower())
    return df

# Find columns that have matching items, even if the feature label is different


def get_matching_columns(df1, df2):
    df1 = preprocess_df(df1)
    df2 = preprocess_df(df2)
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
