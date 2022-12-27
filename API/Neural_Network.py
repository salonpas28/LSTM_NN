import numpy as np
from LSTM import full_clean
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from flasgger import swag_from
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flask import request
import re
import pickle

from flask import Flask, jsonify
import pandas as pd
import joblib

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

vectorizer = open('feature.pkl', 'rb')
load_vectorizer = pickle.load(vectorizer)
vectorizer.close()
model = joblib.load('model_neural.pkl')


@app.route('/')
def hello_world():
    return ("Hello World")


@swag_from("docs/model_neural.yml", methods=['POST'])
@app.route('/neural_network', methods=['POST'])
def NN():

    text = request.form.get('text')
    cleanse_text = [full_clean(text)]

    count_vect = load_vectorizer.transform(cleanse_text)
    prediction = model.predict(count_vect)[0]

    return prediction


@swag_from("docs/model_neural_file.yml", methods=['POST'])
@app.route('/neural_network', methods=["POST"])
def data_clean():

    df = pd.read_csv(request.files.get("file"))

    df_list = []
    for i, k in df.iterrows():
        df_list.append(full_clean(k['Tweet']))
        df['Tweet_cleansed'] = df_list

    # convert list to dataframe
    twt = pd.DataFrame(df_list, columns=['twt'])
    twt = twt.rename(columns={'twt': 'tweets'})
    twt = twt.drop_duplicates()
    twt = twt.drop(index=182).reset_index(drop=True)

    result1 = []
    result2 = []
    df = twt['tweets']
    for i in df:
        count_vect = load_vectorizer.transform(df)
        prediction = model.predict(count_vect)[0]
        result1.append(i)
        result2.append(prediction)
    data = pd.DataFrame({'Text': result1, 'Sentiment': result2})

    return str(data)


if __name__ == '__main__':
    app.run(debug=True)
