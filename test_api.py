from flask import Flask
from flask import request
import random
import pandas as pd

api = Flask(__name__)
api.config["DEBUG"] = True

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.drop('TARGET', axis=1)
    return data

DATA_URL = "./Docs_projet7/df_explore.csv"
df = load_data(80000)

def make_feats():
    key_feat = [col for col in random.sample(df.columns.to_list(), 10)]
    feature_importance = {key : round(random.random(),2) for key in key_feat}
    return feature_importance

@api.route("/predict")
def predict():
    if 'id' in request.args:
        id_client = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    prediction = round(random.random(),2)
    score_metier = round(random.uniform(0,10),2)
    return {'prediction': prediction, 
            'score': score_metier, 
            'id' : id_client}

@api.route("/feats")
def feat_import():
    if 'id' in request.args:
        id_client = int(request.args['id'])
    else : 
        return "Error: No id field provided. Please specify an id."
    feature_importance = make_feats()
    return feature_importance

api.run()