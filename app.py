import os
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from dotenv import load_dotenv
import boto3


from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)

app = Flask(__name__)


class Predictor:
    def __init__(self):
        load_dotenv()
        
        self.nb_model = self.load_model_from_s3('nb_model.pkl')
        self.svm_model = self.load_model_from_s3('svm_model.pkl')
        self.vectorizer = self.load_model_from_s3('vectorizer.pkl')
        
        # Load the RNN model
        # with open('models/rnn_model.pkl', 'rb') as file:
        #    (
        #        self.rnn_model,
        #        self.rnn_vectorizer,
        #        self.rnn_label_encoder,
        #        self.rnn_tokenizer,
        #        self.rnn_max_len,
        #    ) = pickle.load(file)
        
    def load_model_from_s3(self, file_key):
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

    def predict(self, texts):
        # Preprocess the texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize the texts for Naive Bayes and SVM
        vectorized_texts = self.vectorize_text(preprocessed_texts)

        # Make predictions with Naive Bayes and SVM
        nb_predictions = self.nb_model.predict(vectorized_texts)
        svm_predictions = self.svm_model.predict(vectorized_texts)

        # Vectorize the texts for RNN
        # rnn_vectorized_texts = self.vectorize_text_rnn(preprocessed_texts)

        # Make predictions with RNN
        # rnn_predictions = self.predict_rnn(rnn_vectorized_texts)
        # print(rnn_predictions)

        # Convert the predictions to a dictionary format
        response = {
            'Naive Bayes': nb_predictions.tolist(),
            'SVM': svm_predictions.tolist(),
            # 'RNN': rnn_predictions,
        }

        return response

    def preprocess_text(self, text):
        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Join the tokens back into a string
        processed_text = ' '.join(tokens)

        return processed_text

    def vectorize_text(self, text_data):
        X_vec = self.vectorizer.transform(text_data)
        return X_vec

    # def vectorize_text_rnn(self, text_data):
    #    sequences = self.rnn_tokenizer.texts_to_sequences(text_data)
    #    sequences_padded = pad_sequences(sequences, maxlen=self.rnn_max_len)
    #    return sequences_padded

    # def predict_rnn(self, text_data):
    #    predictions = self.rnn_model.predict(text_data)
    #    predictions_labels = np.round(predictions).flatten()
    #    return predictions_labels.tolist()

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))
   
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    texts = data['texts']
    predictions = Predictor.predict(texts)
    return jsonify(predictions)

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
