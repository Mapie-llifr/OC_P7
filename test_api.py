from flask import Flask
from flask import request
import random

api = Flask(__name__)
api.config["DEBUG"] = True

@api.route("/predict")
def predict():
    if 'id' in request.args:
        id_client = int(request.args['id'])
    else:
        id_client = 42
        return "Error: No id field provided. Please specify an id."
    prediction = random.random()
    score_metier = random.uniform(0,10)
    return {'prediction': prediction, 
            'score': score_metier, 
            'id' : id_client}

api.run()