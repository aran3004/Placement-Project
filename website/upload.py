from flask import Blueprint, flash, render_template, request, redirect, url_for, session, jsonify, make_response
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from .models import Datasets, Features
from . import db
import json
import pandas as pd
from .log_reg import logistic_regression_model
from .linear_reg import linear_regression_model

upload = Blueprint('uploads', __name__)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@upload.route('/upload_page')
@login_required
def upload_page():
    return render_template('upload.html', user=current_user)


@upload.route('/upload_dataset', methods=['POST', 'GET'])
@login_required
def upload_dataset():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            session['filename'] = filename
            new_filename = f"{filename.split('.')[0]}_{str(datetime.now())}.csv"
            file.save(os.path.join("datasets", new_filename))
            file_path = os.path.join("datasets", new_filename)
            session['uploaded_data_file_path'] = file_path

        # response = make_response(redirect(url_for('uploads.describe_dataset')))
        # response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return make_response(redirect(url_for('uploads.describe_dataset')))
    return render_template('upload_dataset.html', user=current_user)


@upload.route('/describe_dataset', methods=['POST', 'GET'])
@login_required
def describe_dataset():
    df = pd.read_csv(session['uploaded_data_file_path'])
    if request.method == 'POST':

        dataset_name = request.form.get('dataset_name')
        dataset_task = request.form.get('dataset_task')
        model_type = request.form.get('model_type')
        target = request.form.get('target')
        public_bid = request.form.get('public_bid')
        tags = request.form.get('tags')

        if model_type == 'regression':
            model, rmse = linear_regression_model(df, target=target)
            result = rmse
        elif model_type == 'classification':
            model, accuracy = logistic_regression_model(df, target=target)
            result = accuracy

        new_dataset = Datasets(dataset_name=dataset_name, task=dataset_task,
                               target=target, public_bid=public_bid, tags=tags, user_id=current_user.id, model_type=model_type, file_path=session['uploaded_data_file_path'], result=result)
        db.session.add(new_dataset)
        db.session.commit()
        flash('Dataset uploaded', category='success')
        return redirect(url_for('email.upload_dataset_success_email'))

    return render_template('describe_dataset.html', user=current_user, filename=session['filename'], data=df.head().to_html(classes='mystyle'), columns=df.columns)


@upload.route('/upload_feature', methods=['POST', 'GET'])
@login_required
def upload_feature():
    if request.method == 'POST':

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            session['filename'] = filename
            new_filename = f"{filename.split('.')[0]}_{str(datetime.now())}.csv"
            file.save(os.path.join("features", new_filename))
            file_path = os.path.join("features", new_filename)
            session['uploaded_data_file_path'] = file_path

        return redirect(url_for('uploads.describe_feature'))
    return render_template('upload_feature.html', user=current_user)


@upload.route('/describe_feature', methods=['POST', 'GET'])
@login_required
def describe_feature():
    df = pd.read_csv(session['uploaded_data_file_path'])
    if request.method == 'POST':

        feature_name = request.form.get('feature_name')
        feature_info = request.form.get('feature_info')
        tags = request.form.get('tags')

        new_feature = Features(feature_name=feature_name, info=feature_info, tags=tags,
                               user_id=current_user.id, file_path=session['uploaded_data_file_path'])
        db.session.add(new_feature)
        db.session.commit()
        flash('Feature uploaded', category='success')
        return redirect(url_for('email.upload_feature_success_email'))
        # return redirect(url_for('uploads.profile'))

    return render_template('describe_feature.html', user=current_user, filename=session['filename'], data=df.head().to_html(classes='mystyle'), columns=df.columns)


@upload.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@upload.route('/delete-dataset', methods=['POST'])
def delete_dataset():
    dataset = json.loads(request.data)
    dataset_id = dataset['datasetId']
    dataset = Datasets.query.get(dataset_id)
    if dataset:
        if dataset.user_id == current_user.id:
            os.remove(dataset.file_path)
            db.session.delete(dataset)
            db.session.commit()


@upload.route('/delete-feature', methods=['POST'])
def delete_featuret():
    feature = json.loads(request.data)
    feature_id = feature['featureId']
    feature = Features.query.get(feature_id)
    if feature:
        if feature.user_id == current_user.id:
            os.remove(feature.file_path)
            db.session.delete(feature)
            db.session.commit()

    return jsonify({})
