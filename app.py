import os
from langdetect import detect
# from bussiness.predictor import Predictor

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)

app = Flask(__name__)


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
   
# @app.route('/api/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
#     texts = data['texts']
#     predictions = predictor.predict(texts)
#     return jsonify(predictions)

# @app.route('/api/detect-language', methods=['POST'])
# def detect_language():
#     data = request.get_json(force=True)
#     text = data['text']
#     language = detect(text)
#     is_english = language == 'en'
#     response = {'is_english': is_english}
#     return jsonify(response)


if __name__ == '__main__':
   app.run()
