from flask import Blueprint, session, redirect, url_for
from . import db
from flask_login import current_user
from .models import Datasets, Features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import pandas as pd

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
                    feature_description = feature_descriptions[feature_id]
                    results[f"Dataset ID: {dataset_id}"][f'Feature_ID: {feature_id}'] = {
                        'Description': feature_description, 'Similarity': str(similarity)}
                    print(
                        f"- Feature ID: {feature_id}, Description: {feature_description}, Similarity: {similarity}")

                    dataset_file_path, model_type = get_dataset_by_id(
                        dataset_id)
                    feature_file_path = get_feature_by_id(feature_id)

                    dataset_df = pd.read_csv(dataset_file_path)
                    feature_df = pd.read_csv(feature_file_path)
                    dataset_df = preprocess_df(dataset_df)
                    feature_df = preprocess_df(feature_df)
                    # print(dataset_df.head())
                    # print(feature_df.head())
                    matching_columns = get_matching_columns(
                        dataset_df, feature_df)
                    print(f'{matching_columns}')

            if len(results[f"Dataset ID: {dataset_id}"]) == 0:
                print("No available features for this dataset. Try adding more tags or adjusting the description. Visit your profile to do so.")

            print("-----")

        session['results'] = results

        return redirect(url_for('email.feature_match'))


# Preprocess descriptions (cleaning and tokenization)
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

            # Matching rate above 70%
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
