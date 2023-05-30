import os
import nltk
from langdetect import detect
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import string
from dotenv import load_dotenv
import boto3
import numpy as np

from flask import Flask, redirect, render_template, request, send_from_directory, url_for, jsonify

app = Flask(__name__)

def download_nltk_resources():
    nltk_resources = ['punkt', 'stopwords', 'wordnet']
    for resource in nltk_resources:
        nltk.download(resource)

def load_model_from_s3(file_key):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_bucket_name = os.getenv('AWS_BUCKET_NAME')
        
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

def vectorize_text_rnn(text_data, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(text_data)
    sequences_padded = pad_sequences(sequences, maxlen=max_len)
    return sequences_padded

def predict_rnn(model, text_data):
    predictions = model.predict(text_data)
    predictions_labels = np.round(predictions).flatten()
    return predictions_labels.tolist()


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
    
    rnn_vectorizer = load_model_from_s3('vectorizer_rnn.pkl')
    rnn_model = load_model_from_s3('rnn_model.pkl')
    
    rnn_tokenizer = load_model_from_s3('tokenizer.pkl')
    rnn_max_len = load_model_from_s3('max_len.pkl')
    
    preprocessed_texts = [preprocess_text(text) for text in texts]
    vectorized_texts = vectorize_text(preprocessed_texts, vectorizer)
    
    nb_predictions = nb_model.predict(vectorized_texts).tolist()
    svm_predictions = svm_model.predict(vectorized_texts).tolist()
    
    rnn_vectorized_texts = vectorize_text_rnn(preprocessed_texts, rnn_tokenizer, rnn_max_len)
    rnn_predictions = predict_rnn(rnn_model, rnn_vectorized_texts)
    print(rnn_predictions)
    
    response = {
        'Naive Bayes': nb_predictions,
        'SVM': svm_predictions,
        'RNN': rnn_predictions,
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
