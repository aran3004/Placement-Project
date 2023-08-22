from flask import Blueprint, flash, render_template, request, redirect, url_for
from .models import User, Log
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint('auth', __name__)


@auth.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully', category='success')
                login_user(user, remember=True)
                return redirect(url_for('uploads.profile'))
            else:
                flash('Incorrect password, try again', category='error')
        else:
            flash("Account doesn't exist", category='error')
            return redirect(url_for('auth.sign_up'))

    return render_template('login.html', user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['POST', 'GET'])
def sign_up():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        email = request.form.get('email')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already exists", category='error')
        elif len(email) < 4:
            flash('Email must be greater than 4 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords dont match', category='error')
        elif len(password1) < 7:
            flash('Password must be greater than 7 character.', category='error')
        else:
            new_user = User(email=email, first_name=first_name,
                            password=generate_password_hash(password1, method='sha256'), credit=0)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            new_log = Log(description='Account Created!',
                          user_id=current_user.id)
            print(new_user.id)
            db.session.add(new_log)
            db.session.commit()

            flash("Account created", category='success')
            return redirect(url_for('email.registration_successful'))
    return render_template('sign_up.html', user=current_user)
