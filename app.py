import os
import nltk
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from dotenv import load_dotenv
import boto3

from flask import Flask, redirect, render_template, request, send_from_directory, url_for, jsonify

app = Flask(__name__)

def download_nltk_resources():
    nltk_resources = ['punkt', 'stopwords', 'wordnet']
    for resource in nltk_resources:
        nltk.download(resource)

def load_model_from_s3(file_key):
    aws_access_key_id = "AKIARVGPJVYVKFVELC5U"
    aws_secret_access_key = "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"
    aws_bucket_name = "bucketeer-073738ea-e0e7-4732-9004-a6aa18379209"
        
    s3 = boto3.client('s3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    response = s3.get_object(Bucket=aws_bucket_name, Key=file_key)
    model_data = response['Body'].read()
    return pickle.loads(model_data)

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    tokens = [token for token in tokens if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def vectorize_text(text_data, vectorizer):
    X_vec = vectorizer.transform(text_data)
    return X_vec

@app.before_request
def setup():
    load_dotenv()
    download_nltk_resources()

@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
    name = request.form.get('name')

    if name:
        print('Request for hello page received with name=%s' % name)
        return render_template('hello.html', name=name)
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return redirect(url_for('index'))
   
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    texts = data['texts']
    
    nb_model = load_model_from_s3('nb_model.pkl')
    svm_model = load_model_from_s3('svm_model.pkl')
    vectorizer = load_model_from_s3('vectorizer.pkl')
    
    preprocessed_texts = [preprocess_text(text) for text in texts]
    vectorized_texts = vectorize_text(preprocessed_texts, vectorizer)
    
    nb_predictions = nb_model.predict(vectorized_texts).tolist()
    svm_predictions = svm_model.predict(vectorized_texts).tolist()
    
    response = {
        'Naive Bayes': nb_predictions,
        'SVM': svm_predictions,
    }
    
    return jsonify(response)

@app.route('/api/detect-language', methods=['POST'])
def detect_language():
    data = request.get_json(force=True)
    text = data['text']
    language = detect(text)
    is_english = language == 'en'
    response = {'is_english': is_english}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
