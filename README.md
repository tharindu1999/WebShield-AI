# WebShield AI: Website Risk Classifier

WebShield AI is a machine learning based web application that predicts whether a website is legitimate or phishing using features extracted from its URL and HTML content.

The application was developed using Python, scikit learn and Streamlit. It is publicly deployed using Streamlit Community Cloud and is also available as a Docker container.

Live application:

https://webshield-ai-tn8mtu7syclccvmrqxu9vx.streamlit.app/

## 1. Problem Statement

Phishing websites often imitate legitimate websites in order to collect sensitive information such as usernames, passwords and payment details.

Since phishing pages can look similar to genuine websites, identifying them through visual inspection alone can be difficult.

The aim of WebShield AI is to use machine learning to analyse measurable website characteristics and predict whether a website is legitimate or phishing.

## 2. Use Case

The user enters a public website URL and selects Analyze Website.

The application automatically extracts the required website features and displays:

1. Predicted class

2. Phishing probability

3. Legitimate probability

4. Risk level

5. Features provided to the model

6. Model evaluation results

The prediction is intended as an initial risk assessment and should not be treated as proof that a website is safe or malicious.

## 3. Solution Overview

The application follows a simple process.

Website URL → Feature Extraction → Random Forest Model → Prediction → Streamlit Dashboard

The URL and HTML content are analysed to extract the same features used during model training.

These values are passed to a trained Random Forest classifier which predicts whether the website is legitimate or phishing.

The application does not require a database or separate backend service.

## 4. Dataset

The project uses the PhiUSIIL Phishing URL Website Dataset from the UCI Machine Learning Repository.

Dataset ID: 967

Original records: 235,795

Records after duplicate URL removal: 235,370

Original label 1 represents Legitimate.

Original label 0 represents Phishing.

For this project the labels were converted to:

0 represents Legitimate

1 represents Phishing

Dataset source:

https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Dataset citation:

Prasad, A. and Chandra, S. (2024). PhiUSIIL Phishing URL Website Dataset. UCI Machine Learning Repository.

https://doi.org/10.1016/j.cose.2023.103545

## 5. AI and Machine Learning Approach

The application uses a RandomForestClassifier from scikit learn.

The dataset was divided into approximately 80 percent training data and 20 percent testing data using a stratified split.

Training samples: 188,296

Testing samples: 47,074

The model uses 20 features that can also be extracted from a live website. These include URL length, domain length, HTTPS usage, page title, description, form fields, images, JavaScript files, CSS files, iframes and link information.

The final model achieved:

Accuracy: 99.89 percent

Precision: 99.89 percent

Recall: 99.86 percent

F1 Score: 99.88 percent

ROC AUC: 0.99996

These results represent performance on the test portion of the PhiUSIIL dataset. Performance on live websites may vary because real websites and phishing techniques can differ from the training data.

The trained model is stored in:

`models/web_risk_model.pkl`

Model information is stored in:

`models/model_metadata.json`

## 6. Application Architecture

The application uses a simple architecture.

User URL → Streamlit → Feature Extractor → Random Forest Model → Prediction

The main files are:

`app.py` contains the Streamlit interface and prediction flow.

`feature_extractor.py` handles website retrieval and feature extraction.

`feature_definitions.py` contains the ordered model features.

`train.py` handles dataset preparation, model training and evaluation.

`models/` contains the trained model and model metadata.

## 7. Technology Stack

Python

Streamlit

pandas

NumPy

scikit learn

joblib

requests

Beautiful Soup

pytest

Docker

Streamlit Community Cloud

## 8. Local Setup Instructions

Python 3.11 is recommended.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open:

http://localhost:8501

Run tests using:

```bash
python -m pytest -q
```

The application loads the saved model when it starts and does not retrain it during normal use.

## 9. Deployment Details

The application is deployed using Streamlit Community Cloud.

Live application:

https://webshield-ai-tn8mtu7syclccvmrqxu9vx.streamlit.app/

The deployment is connected to the GitHub repository and uses `app.py` as the application entry point.

Dependencies are installed from `requirements.txt`, while the trained model and metadata are loaded from the `models` directory.

The application is also Dockerized for portable deployment.

## 10. Web Application Usage

To use WebShield AI:

1. Open the application.

2. Enter a complete public URL such as `https://example.com`.

3. Select Analyze Website.

4. Review the predicted class and probabilities.

5. View the extracted model features and model evaluation information if required.

Only public HTTP and HTTPS websites are accepted.

Localhost and private network addresses are rejected.

## 11. Docker Instructions

Build the Docker image:

```bash
docker build -t webshield-ai:1.0 .
```

Run the container:

```bash
docker run --rm -p 8501:8501 --name webshield-ai webshield-ai:1.0
```

Open:

http://localhost:8501

The Docker image contains the Streamlit application, feature extraction code, trained model and required dependencies.
