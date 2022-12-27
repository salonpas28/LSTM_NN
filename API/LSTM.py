import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from flasgger import swag_from
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flask import request
import re
import pickle

from LSTM import full_clean
from LSTM import X
from LSTM import tokenizer

from flask import Flask, jsonify
import pandas as pd

df_data = pd.read_csv('data.csv', encoding='latin')

app = Flask(__name__)


app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling')
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

sentiment = ['negatif', 'netral', 'positif']

file_tokenizer = open('tokenizer.pickle', 'rb')
# max_features = 100000
# tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
file_sequencer = open('x_pad_sequences.pickle', 'rb')

load_tokenizer = pickle.load(file_tokenizer)
load_sequencer = pickle.load(file_sequencer)
file_sequencer.close()
model_lstm = load_model('model_sentiment.h5')


@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():

    result1 = []
    result2 = []
    text = request.form.get('text')
    sentiment = ['negative', 'neutral', 'positive']

    cleanse_text = [full_clean(text)]
    predicted = tokenizer.texts_to_sequences(cleanse_text)
    guess = pad_sequences(predicted, maxlen=X.shape[1])

    model = load_model('model_sentiment.h5')
    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])
    result1.append()
    result2.append(sentiment[polarity])
    df_df = pd.DataFrame({'Text': result1, 'Sentiment': result2})

    json_response = {
        'status_code': 200,
        'description': "Original text",
        'sentiment': df_df
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm_file', methods=[''])
def lstm_file():

    df = pd.read_csv(request.files.get("file"))

    df_list = []
    for i, k in df.iterrows():
        df_list.append(full_clean(k['Tweet']))
        df_data['Tweet_cleansed'] = df_list

    # convert list to dataframe
    twt = pd.DataFrame(df_list, columns=['twt'])
    twt = twt.rename(columns={'twt': 'tweets'})
    twt = twt.drop_duplicates()
    twt = twt.drop(index=182).reset_index(drop=True)

    result1 = []
    result2 = []
    df_data1 = twt['tweets']
    sentiment = ['positive', 'neutral', 'negative']

    for i in df_data1:
        predicted = tokenizer.texts_to_sequences(i)
        guess = pad_sequences(predicted, maxlen=X.shape[1])

        model = load_model('model_sentiment.h5')
        prediction = model.predict(guess)
        polarity = np.argmax(prediction[0])
        result1.append(i)
        result2.append(sentiment[polarity])

    df_df = pd.DataFrame({'Text': result1, 'Sentiment': result2})

    json_response = {
        'status_code': 200,
        'description': 'Original Text',
        'sentiment': df_df
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run(debug=True)
