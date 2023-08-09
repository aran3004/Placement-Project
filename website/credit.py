from flask import Blueprint, redirect, url_for, request, flash
from flask_login import current_user, login_required
from .models import User
from . import db

credit = Blueprint('credit', __name__)


@credit.route('/add_credit', methods=['GET', 'POST'])
@login_required
def add_credit():
    if request.method == 'POST':
        credit_to_add = request.form.get('add_credit')
        print(f'Credit to be added: {credit_to_add}')
        current_user.credit = int(current_user.credit) + int(credit_to_add)
        db.session.commit()
        flash('Credit Added', category='success')
    return redirect(url_for('uploads.profile'))


@credit.route('/withdraw_credit', methods=['GET', 'POST'])
@login_required
def withdraw_credit():
    if request.method == 'POST':
        credit_to_withdraw = request.form.get('withdraw_credit')
        new_credit = int(current_user.credit) - int(credit_to_withdraw)
        if new_credit < 0:
            flash('Trying to withdraw too much credit', category='warning')
        else:
            current_user.credit = new_credit
            current_user.credit
            db.session.commit()
            flash('Credit Withdrawn', category='success')
    return redirect(url_for('uploads.profile'))
