from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from flask_mail import Mail

db = SQLAlchemy()
DB_NAME = 'database.db'
mail = Mail()


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'ASOUHsdoigjh u'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = 'aranmahal2001@gmail.com'
    app.config['MAIL_PASSWORD'] = 'jsibtvargxxvixas'
    db.init_app(app)
    mail.init_app(app)

    from .views import views
    from .auth import auth
    from .upload import upload
    from .email import email
    from .search_algo import search_algo

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(upload, url_prefix='/')
    app.register_blueprint(email, url_prefix='/')
    app.register_blueprint(search_algo, url_prefix='/')

    from .models import User, Datasets

    with app.app_context():
        create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('instance/'+DB_NAME):
        db.create_all()
        print('Created Database')
