# Collaborative Data Markets Platform
This project demonstartes a fully working data markets platform where clients are able to upload current data or ML problem they have. Other clients are able to view all ML models and choose to upload their own data towards those problems. If their data results in an improvement in model accuracy, model contributors are rewarded according to their data feature importance in the model (based on Shapley Values). This code also leverages NLP to help match clients data to suitable tasks and provide reward.

The code is created primarily using the Flask framework to communicate between the frontend and backend operations.

# Project Structure:
**datasets, features**: Contain all the csv files that are used for model creation and comparison, aswell as by the NLP algorithm. <br />
**instance/database.db**: Initialise and stores the database that hold all user info for authentication and associated data and rewards. <br />
**website/static**: Contains all styling for front end, email and JS<br />
**website/__init__.py**: Initialise database and Flask app<br />
**website/auth.py**: Contains functionality and routing for login and signout<br />
**website/classification.py**: Modules for classification<br />
**website/credit.py**: Contains functionality and routing for credit deposit/withdrawal and transfer<br />
**website/email.py**: Email information for signup and feature matches<br />
**website/linear_reg.py**: Contains linear regression model<br />
**website/log_reg.py**: Contains logisitic regression model<br />
**website/models.py**: SQL Database models: features, datasets, users and logs<br />
**website/regression.py**: Contains Shapley calculations<br />
**website/search_algo.py**: NLP algorithm to search for matching features to current ML models<br />
**website/upload.py**: Contains Flask routing and functionality for uploading a dataset or feature<br />
**website/views.py**: Setting the blueprint to other Flask files<br />
**environment.yaml**: Contains dependencies<br />
**main.py**: Create app instance<br />
**requirements.txt**: Contains all requirements<br />

# Installation and Running the app
Use the 'requirements.txt' file to install all the dependencies

To run the app, run "python app.py" in the terminal
