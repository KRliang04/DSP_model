#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:15:10 2019

@author: kiraliang
"""
import pickle
import pandas as pd

MODEL_PATH = "./model/LR_model.sav"
THRESHOLD = 50

def load_model():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    
    return model

def predict(model,df):
    y_pred = model.predict_proba(df)
    dis_score = round(y_pred[0][0]*100,2)
    sat_score = round(y_pred[0][1]*100,2)
    if dis_score > THRESHOLD:
        print('{"dissatisfied":"' + str(dis_score) + '"}')
    else:
        print('{"satisfied":"' + str(sat_score) + '"}')
def main():
    model = load_model()
    features_df = pd.read_csv("features/features_predict.csv").iloc[:,1:]
    predict(model,features_df)
    
if __name__ == "__main__":
    main()