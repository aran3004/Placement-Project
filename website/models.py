from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Datasets(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(150))
    task = db.Column(db.String(10000))
    # training and test data split
    # earliest date, time intervals, end date for matching datasets
    file_path = db.Column(db.String(10000))
    model_type = db.Column(db.String(1000))
    target = db.Column(db.String(100))
    public_bid = db.Column(db.Integer)
    tags = db.Column(db.String(1000))
    # create another db for tags later on
    result = db.Column(db.Float(100))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class Features(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feature_name = db.Column(db.String(150))
    info = db.Column(db.String(10000))
    file_path = db.Column(db.String(10000))
    tags = db.Column(db.String(1000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    first_name = db.Column(db.String(150))
    password = db.Column(db.String(150))
    credit = db.Column(db.String(150))
    datasets = db.relationship('Datasets')
    features = db.relationship('Features')
