from flask import Flask
from flask import request
#import random
import pandas as pd
from joblib import load

api = Flask(__name__)
api.config["DEBUG"] = True

@api.route("/")
def hello_world():
    return "Hello World"

MODEL = 'small_model_final.joblib'
#DATA_URL = "https://github.com/Mapie-llifr/OC_P7/blob/main/Docs_projet7/small_df_model_final.csv" #"https://eu.pythonanywhere.com/user/Mapiellifr/files/home/Mapiellifr/OC_P7/small_df_model_final.csv"
MODEL_URL = "https://github.com/Mapie-llifr/OC_P7/blob/main/" + MODEL
#MODEL = "pipeline_lightGBM_final.joblib"        #local
DATA_URL = "./Docs_projet7/small_df_model_final.csv"  #local
#MODEL_URL = MODEL                               # local
SEUIL = 0.20


def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.drop('TARGET', axis=1)
    return data


def prediction(client):
    X_client = df[df['SK_ID_CURR'] == client]
    X_client = X_client.drop('SK_ID_CURR', axis=1)
    y_pred = clf.predict_proba(X_client)[0,1]
    return y_pred


def accord(pred) : 
    if pred < SEUIL :
        pret = 1
    elif pred < 0.5 : 
        pret = 5
    else : 
        pret = 0
    return pret


def make_feats(client):
    
    X_client = df[df['SK_ID_CURR'] == client]
    X_client = X_client.drop('SK_ID_CURR', axis=1)
    
    if MODEL == "pipeline_lightGBM_final.joblib":
        feats_pred = clf.predict_proba(X_client, pred_contrib=True)
        importance_serie = pd.Series(data=feats_pred[0,0:-1], 
                                 index=X_client.columns)
        
    elif MODEL == "small_model_final.joblib":
        scaler = clf.named_steps['classifier'].named_steps['scaler']
        values = scaler.transform(X_client)
        feats_pred = clf.named_steps['classifier'].named_steps['lr'].coef_ * values
        importance_serie = pd.Series(data=feats_pred[0], 
                                     index=X_client.columns)
    
    best_feats = importance_serie.loc[importance_serie.abs().sort_values(
                                                ascending=False)[:10].index]
    feature_importance = best_feats.to_dict()
    return feature_importance

df = load_data(3000)

clf = load(MODEL_URL)

@api.route("/predict")
def predict():
    if 'id' in request.args:
        id_client = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."
    pred_val = prediction(id_client)
    score_metier = accord(pred_val)
    return {'prediction': pred_val, 
            'score': score_metier, 
            'id' : id_client}

@api.route("/feats")
def feat_import():
    clf = load(MODEL_URL)
    if 'id' in request.args:
        id_client = int(request.args['id'])
    else : 
        return "Error: No id field provided. Please specify an id."
    feature_importance = make_feats(id_client)
    return feature_importance

api.run()

# python api.py
