from flask import Blueprint, render_template, flash, redirect, url_for, session
from flask_login import current_user
from flask_mail import Message
from . import mail

email = Blueprint('email', __name__)


@email.route('/registration_successful')
def registration_successful():
    sender = 'noreply@app.com'
    msg_title = 'Account Created Successfully'
    msg = Message(msg_title, sender=sender, recipients=[current_user.email])
    dataset_link = url_for('uploads.upload_dataset', _external=True)
    feature_link = url_for('uploads.upload_feature', _external=True)
    msg.html = render_template(
        'registered_email.html', dataset_link=dataset_link, feature_link=feature_link)
    try:
        mail.send(msg)
        flash('Email sent', category='success')
        return redirect(url_for('views.home'))
    except Exception as e:
        print(e)
        return f'the email was not sent {e}'


@email.route('/upload_dataset_success_email')
def upload_dataset_success_email():
    sender = 'noreply@app.com'
    msg_title = 'Dataset Uploaded Successfully'
    msg = Message(msg_title, sender=sender, recipients=[current_user.email])
    dataset_data = current_user.datasets[-1]
    filename = session['filename']
    link = url_for('search_algo.search', _external=True)
    msg.html = render_template(
        'dataset_email.html', dataset=dataset_data, filename=filename, link=link)
    session.clear()
    try:
        mail.send(msg)
        flash('Email sent', category='success')
        return redirect(url_for('uploads.profile'))
    except Exception as e:
        print(e)
        return f'the email was not sent {e}'


@email.route('/upload_feature_success_email')
def upload_feature_success_email():
    sender = 'noreply@app.com'
    msg_title = 'Feature Uploaded Successfully'
    msg = Message(msg_title, sender=sender, recipients=[current_user.email])
    feature_data = current_user.features[-1]
    filename = session['filename']
    link = url_for('uploads.profile', _external=True)
    msg.html = render_template(
        'feature_email.html', dataset=feature_data, filename=filename, link=link)
    session.clear()
    try:
        mail.send(msg)
        flash('Email sent', category='success')
        return redirect(url_for('uploads.profile'))
    except Exception as e:
        print(e)
        return f'the email was not sent {e}'


@email.route('/feature_match')
def feature_match():
    sender = 'noreply@app.com'
    msg_title = 'We have Matched a Feature to your Dataset!'
    msg = Message(msg_title, sender=sender, recipients=[current_user.email])
    results = session['results']

    msg.html = render_template('feature_match_email.html', results=results)
    session.clear()
    try:
        mail.send(msg)
        flash('Email sent', category='success')
        return redirect(url_for('uploads.profile'))
    except Exception as e:
        print(e)
        return f'the email was not sent {e}'
